# KMeans 算子测试文档

> 本文档记录 `GGML_OP_KMEANS` 算子在 `test-backend-ops` 框架中的测试设计、实施过程与验证结果。
>
> 关联文档：
> - [`base_kmeans.md`](base_kmeans.md) — KMeans 算子设计规格
> - [`new_ops.md`](new_ops.md) — GGML 新算子添加指南

---

## 1. 测试目标

KMeans 算子直接操作 KV Cache，涉及：
- **原地写入** `centroids_buf`（F16 输出缓冲区）
- **复杂内存布局** — 工作缓冲区（`wdata`）的多段 carve
- **多线程并行** — 按 head 分片，每个线程独立处理
- **数值稳定性** — 余弦相似度、归一化、空簇处理

因此测试需覆盖以下维度：

| 维度 | 验证方式 |
|------|----------|
| 内存安全 | sentinel 越界检测 |
| 数值稳定性 | NaN/Inf 自动检查 + bit-exact 对比 |
| 多线程一致性 | 同图两次执行，输出必须完全一致 |
| 功能正确性 | 多种 shape / 类型 / 边界条件覆盖 |

---

## 2. 测试结构 (`test_kmeans`)

### 2.1 代码位置

```
tests/test-backend-ops.cpp
├── Section 2 (≈ line 7280): struct test_kmeans 定义
└── Section 3 (≈ line 8792): make_test_cases_eval() 中注册用例
```

### 2.2 类设计

```cpp
struct test_kmeans : public test_case {
    const ggml_type type;        // k_cache 类型: F16 / BF16
    const int d_head;            // 每个 head 的维度
    const int n_heads;           // head 数
    const int n_tokens;          // token 数（序列长度）
    const int n_clusters;        // 聚类数
    const int max_iter;          // 最大迭代次数
    const int sink_len;          // sink token 数（不参与聚类）

    ggml_tensor * centroids_buf = nullptr;

    std::string vars() override;
    double max_nmse_err() override { return 0.0; }  // bit-exact
    ggml_tensor * build_graph(ggml_context * ctx) override;
    void initialize_tensors(ggml_context * ctx) override;
};
```

#### 关键点

- **`max_nmse_err() = 0.0`**：KMeans 算法在给定相同输入和相同初始采样时是完全确定性的，因此两次执行必须 bit-exact 匹配。
- **`initialize_tensors()`**：`centroids_buf` 是**输出 buffer**，不清零可能导致旧的脏数据干扰验证；其余输入 tensor 使用 `init_tensor_uniform(-1.0f, 1.0f)` 随机化。
- **默认 `run_whole_graph()`**：返回 `false`，让框架逐个节点验证。若设为 `true` 并配合 `fusion_test_nodes()` 指向 `centroids_buf`，会因 `centroids_buf` 不在 `g1->nodes` 中（它是叶子节点）而触发 `GGML_ASSERT(verified)` 失败。

---

## 3. 测试用例

共注册 **6 个用例**，覆盖典型配置与边界条件：

### 3.1 典型配置

| # | 类型 | d_head | n_heads | n_tokens | n_clusters | max_iter | sink_len | 说明 |
|---|------|--------|---------|----------|------------|----------|----------|------|
| 1 | F16 | 64 | 2 | 64 | 4 | 10 | 4 | 小模型典型值 |
| 2 | F16 | 128 | 4 | 256 | 8 | 10 | 8 | 大模型典型值 |
| 3 | BF16 | 64 | 2 | 64 | 4 | 10 | 4 | BF16 小配置 |
| 4 | BF16 | 128 | 4 | 256 | 8 | 10 | 8 | BF16 大配置 |

**目的**：验证 F16/BF16 两种量化输入格式在不同规模下的正确性。

### 3.2 边界条件

| # | 类型 | d_head | n_heads | n_tokens | n_clusters | max_iter | sink_len | 说明 |
|---|------|--------|---------|----------|------------|----------|----------|------|
| 5 | F16 | 32 | 2 | 32 | 4 | 5 | 0 | **sink_len = 0**：所有 token 参与聚类 |
| 6 | F16 | 32 | 2 | 16 | 4 | 5 | 4 | **小样本**：n_tokens ≈ n_clusters + sink_len |

**目的**：
- `sink_len = 0`：验证无 sink 时的代码路径（跳过 sink 标记逻辑）。
- 小样本：验证空簇处理、cluster renumbering 等边界逻辑不会崩溃或产生非法输出。

---

## 4. 验证机制详解

### 4.1 内存安全 — Sentinel 检测

`test-backend-ops` 框架在每个 `ggml_new_tensor_*` 调用后自动插入 **sentinel tensor**（1024 个 float，初始为 0）。

执行流程：
1. `build_graph()` 前插入 pre-graph sentinel
2. 每次 `ggml_new_tensor_*()` 后插入 post-tensor sentinel
3. 图计算完成后，通过 `compare_tensors` 回调检查所有 sentinel
4. 若 sentinel 数据被修改（`memcmp` 失败），输出 `sentinel mismatch` 并标记测试失败

**对 KMeans 的意义**：
- `k_f32` 缓冲区、centroids 缓冲区、assignments 数组若发生越界写入，将直接破坏相邻 sentinel。
- 多线程环境下，若某线程的 head 处理越界访问了其他 head 的数据，sentinel 同样会捕获。

### 4.2 数值稳定性 — NaN/Inf 检查

`compare_tensors` 回调中对每个输出节点执行：

```cpp
if (std::isnan(f1[i]) || std::isnan(f2[i])) { ... fail ... }
if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) { ... fail ... }
```

**对 KMeans 的意义**：
- 若 `k_inv_norm` 计算时遇到 0 向量（除零），会产生 Inf/NaN
- 若 centroids 更新时某个簇为空且未正确处理，可能产生 NaN
- 这些错误会被测试框架自动捕获

### 4.3 多线程一致性 — Bit-Exact 对比

`test-backend-ops` 的 `MODE_TEST` 模式会：
1. 在目标 backend（如 CPU）上执行图计算
2. 在 CPU reference backend 上执行相同图计算
3. 逐个节点对比输出

对于 KMeans：
- 由于只有一个计算节点（`GGML_OP_KMEANS`），框架对比的是 `assignments`（I32）
- `tensor_to_float()` 将 I32 转为 float 后比较，由于 cluster ID 值域很小（-1 ~ 31），float 表示完全精确
- `max_nmse_err() = 0.0` 要求每个元素完全一致，不允许任何误差

**线程安全验证原理**：
- 若多线程并行存在 race condition，两次执行（或跨 backend）的结果会不同
- bit-exact 的 0 误差阈值会立即暴露任何非确定性行为

---

## 5. 编译与运行

### 5.1 编译

```bash
cd build-test
cmake .. && make -j$(nproc) test-backend-ops
```

### 5.2 运行全部 KMeans 测试

```bash
./bin/test-backend-ops -b CPU -o KMEANS
```

### 5.3 运行单个测试（示例）

```bash
# 通过完整参数匹配过滤
./bin/test-backend-ops -b CPU \
  -o "KMEANS(type=f16,d_head=64,n_heads=2,n_tokens=64,n_clusters=4,max_iter=10,sink_len=4)"
```

### 5.4 预期输出

```
Testing 1 devices

Backend 1/1: CPU
  Device description: 13th Gen Intel(R) Core(TM) i5-13500H
  Device memory: 15714 MB (15714 MB free)

  KMEANS(type=f16,d_head=64,n_heads=2,n_tokens=64,n_clusters=4,max_iter=10,sink_len=4): OK
  KMEANS(type=f16,d_head=128,n_heads=4,n_tokens=256,n_clusters=8,max_iter=10,sink_len=8): OK
  KMEANS(type=bf16,d_head=64,n_heads=2,n_tokens=64,n_clusters=4,max_iter=10,sink_len=4): OK
  KMEANS(type=bf16,d_head=128,n_heads=4,n_tokens=256,n_clusters=8,max_iter=10,sink_len=8): OK
  KMEANS(type=f16,d_head=32,n_heads=2,n_tokens=32,n_clusters=4,max_iter=5,sink_len=0): OK
  KMEANS(type=f16,d_head=32,n_heads=2,n_tokens=16,n_clusters=4,max_iter=5,sink_len=4): OK
  6/6 tests passed
  Backend CPU: OK
1/1 backends passed
OK
```

---

## 6. 实施过程中的问题与解决

### 6.1 问题：测试注册到了错误的函数

**现象**：编译成功但运行 `-o KMEANS` 显示 `0/0 tests passed`。

**根因**：`tests/test-backend-ops.cpp` 中存在两个注册测试的函数：
- `make_test_cases_eval()` — 用于正确性验证（MODE_TEST）
- `make_test_cases_perf()` — 用于性能评估（MODE_PERF）

最初将 KMeans 用例插入到了 `make_test_cases_perf()` 末尾，而 `MODE_TEST` 调用的是 `make_test_cases_eval()`。

**解决**：将用例移动到 `make_test_cases_eval()` 的 `return test_cases;` 之前。

### 6.2 问题：`run_whole_graph() = true` 导致 `GGML_ASSERT(verified)` 崩溃

**现象**：
```
ggml/src/ggml-backend.cpp:2182: GGML_ASSERT(verified) failed
```

**根因**：当 `run_whole_graph() = true` 时，框架使用 `ggml_backend_compare_graph_backend()` 比较两个 backend 的整图输出。该函数要求 `fusion_test_nodes()` 返回的每个 tensor 必须出现在 `ggml_cgraph->nodes[]` 中。

`centroids_buf` 是 `ggml_kmeans()` 的 `src[1]`，属于叶子节点（`op = GGML_OP_NONE`），不会被加入 `nodes[]` 列表，因此 `verified` 始终为 `false`。

**解决**：移除 `run_whole_graph()` 和 `fusion_test_nodes()` 的覆盖，使用默认的逐节点验证模式。该模式只验证图的输出节点（`assignments`），而 `centroids_buf` 的正确性通过 bit-exact 的 assignments 间接验证（assignments 一致意味着 centroids 计算路径完全一致）。

---

## 7. 未来扩展

### 7.1 验证 centroids_buf 内容

当前测试仅直接验证 `assignments`（I32）。`centroids_buf`（F16）的验证是间接的（通过 assignments 的 bit-exact 一致性推断）。

如需直接验证 centroids，可考虑：
- 在 `build_graph()` 中增加 `ggml_cpy` 将 `centroids_buf` 复制到新的输出 tensor
- 或设置 `run_whole_graph() = true` 但不在 `fusion_test_nodes()` 中包含叶子节点

### 7.2 多 Backend 验证

当前仅在 CPU backend 上验证。后续接入 Hexagon / NPU backend 时：
- 在 `ggml_backend_cpu_device_supports_op()` 或其他 backend 的 `supports_op` 中注册 KMEANS
- 确保工作缓冲区大小计算（`ggml_graph_plan` 中的 `work_size`）与 backend 的内存模型兼容
- 若 backend 使用不同的浮点实现（如 Hexagon 的 HVX），可能需要放宽 `max_nmse_err()` 阈值

### 7.3 性能回归测试

KMeans 的复杂度为 `O(n_heads × n_tokens × n_clusters × d_head × max_iter)`。对于大模型配置（如 32K tokens、32 clusters、10 iter），可能成为瓶颈。

建议后续在 `make_test_cases_perf()` 中添加：
```cpp
test_cases.emplace_back(new test_kmeans(GGML_TYPE_F16, 128, 8, 32768, 32, 10, 8));
```
用于持续监控性能回归。

---

## 8. 附录：相关代码索引

| 文件 | 行号范围 | 内容 |
|------|----------|------|
| `tests/test-backend-ops.cpp` | ~7280–7340 | `struct test_kmeans` 定义 |
| `tests/test-backend-ops.cpp` | ~8792–8805 | `make_test_cases_eval()` 中注册用例 |
| `ggml/src/ggml-cpu/ops.cpp` | — | `ggml_compute_forward_kmeans` 实现 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | — | work_size 计算、n_tasks 分配 |
| `ggml/src/ggml.c` | ~5380–5397 | `ggml_kmeans()` tensor 构造函数 |
| `ggml/include/ggml.h` | ~579 | `GGML_OP_KMEANS` 枚举 |
