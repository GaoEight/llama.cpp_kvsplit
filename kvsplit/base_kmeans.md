# KMeans Cache 聚类算子设计文档

## 1. 背景与目标

在 LLM 推理的 KV Cache 压缩场景中，对 K cache 中的向量按 head 做 KMeans 聚类，将相似的 token 聚为一类，用于后续的 cache 压缩或分组 attention。

## 2. 算子语义

### 2.1 输入

| 输入 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `k_cache` | `[d_head, n_heads, n_tokens]` | F16 / BF16 | K cache，每 slice `k_cache[:, h, t]` 是 head `h` 下 token `t` 的 k 向量 |
| `centroids_buf` | `[d_head, n_heads, n_clusters]` | F32 | 预分配的 centroids 输出 buffer，算子执行时原地写入 |

> **Note**: `n_tokens` 维度在第 2 轴（最后一个维度），每 slice `k_cache[:, h, t]` 是一个 d_head 维向量。
>
> 在 llama.cpp 的 KV cache 中，K 的实际物理 layout 为 `[d_head, n_head_kv, n_kv, ns]`（4D），其中 `n_kv` 是总序列长度，`ns` 是序列数。若 `ns == 1`，可 `reshape_3d` 后传入本算子。

### 2.2 参数（op_params 编码）

| 参数 | op_params 索引 | 类型 | 说明 |
|------|---------------|------|------|
| `n_clusters` | `[0]` | int32 | 聚类数 k（每 head 独立聚类，cluster 数相同） |
| `max_iter` | `[1]` | int32 | KMeans 最大迭代次数 |
| `sink_len` | `[2]` | int32 | 前 `sink_len` 个 token 不参与聚类（保留原始语义） |

### 2.3 输出

| 输出 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `assignments` (dst) | `[n_heads, n_tokens]` | I32 | 每个 token 的聚类分配结果 |
| `centroids` (src[1]) | `[d_head, n_heads, n_clusters]` | F16 | 每 head 每 cluster 的 centroid 向量 |

#### assignments 编码规则：
- `assignments[h, t] = -1`：若 `t < sink_len`（sink token，不参与聚类）
- `assignments[h, t] = c`：若 `t >= sink_len`，其中 `c ∈ [0, n_clusters-1]` 为 cluster ID

#### Cluster 编号规则（关键设计）：
1. 先执行 KMeans 得到初步分配
2. 对每个 cluster，找到其内部最小的 token 索引
3. 按这个最小索引从小到大对所有 cluster 排序
4. 重新编号：最小索引最小的 cluster → ID 0，次小 → ID 1，以此类推
5. centroids buffer 中的顺序与这个重新编号后的顺序一致

> 示例：某 head 聚类后，cluster A 含 tokens {5, 10}（最小=5），cluster B 含 {2, 8}（最小=2），cluster C 含 {7}（最小=7）
> - 排序：B(2) < A(5) < C(7)
> - 重新编号：B→0, A→1, C→2
> - assignments[5]=1, assignments[10]=1, assignments[2]=0, assignments[8]=0, assignments[7]=2
> - centroids[:, h, 0] = centroid of B, centroids[:, h, 1] = centroid of A, centroids[:, h, 2] = centroid of C

### 2.4 聚类算法细节

#### 距离度量

使用 **1 - cosine similarity**：

```
dist(a, b) = 1 - (a · b) / (||a|| * ||b||)
```

等价于对每个 k 向量，分配到使其 **cosine similarity 最大**（即 dot product / norm 最大）的 centroid。

#### 每 head 独立聚类

对每个 head `h`：
1. 取该 head 下 token `[sink_len, n_tokens)` 的 k 向量
2. 在这些向量上运行 KMeans，聚为 `n_clusters` 个 cluster
3. 生成该 head 的 assignments 和 centroids

不同 head 之间的聚类完全独立，可并行。

#### 初始化策略

**均匀采样**：在 sink 之后的 token 中，均匀间隔选取 `n_clusters` 个向量作为初始 centroid。

> 例如：100 个 token（sink=4），取 8 个 cluster，则初始 centroid 取 token idx = 4, 16, 28, 40, 52, 64, 76, 88 对应的 k 向量。

#### 迭代终止条件

仅使用 `max_iter`，**不提前停止**。

#### 空 cluster 处理

不处理。若某次迭代后某 cluster 没有成员，其 centroid 保持上次值不变。

## 3. GGML API 设计

```cpp
// KMeans clustering on K cache
// 参数通过 op_params 编码: [n_clusters, max_iter, sink_len]
//
// k_cache:       [d_head, n_heads, n_tokens], type F16/BF16
// centroids_buf: [d_head, n_heads, n_clusters], type F32 (预分配，算子写入)
//
// 返回 assignments: [n_heads, n_tokens], type I32
GGML_API struct ggml_tensor * ggml_kmeans(
        struct ggml_context * ctx,
        struct ggml_tensor  * k_cache,
        struct ggml_tensor  * centroids_buf,
        int32_t               n_clusters,
        int32_t               max_iter,
        int32_t               sink_len);
```

## 4. 模型集成位置说明

### 接入点

在 `src/models/llama.cpp` 的 `llm_build_llama` 中，或 `src/llama-graph.cpp` 的 `build_attn`（KV cache 版本）中：

```cpp
// store to KV cache
{
    const auto & k_idxs = inp->get_k_idxs();
    const auto & v_idxs = inp->get_v_idxs();
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
}

// === KMeans 接入点 ===
// 此时 k 为完整 KV cache，shape = [d_head, n_head_kv, n_kv, ns]
// 若 ns == 1，可 reshape_3d 后传入
```

### 关于 permute 的影响

**KMeans 算子在 `build_attn_mha` 之前接入，不会受到 permute 影响。**

原因：
- `build_attn_mha` 内部才对 k 做 `ggml_permute(ctx0, k, 0, 2, 1, 3)`
- 在 `build_attn_mha` 之前，k 的 shape 为 `[d_head, n_head_kv, n_kv, ns]`（未 permute）
- KMeans 算子在此位置处理原始 layout 的 k cache，处理完后 `build_attn_mha` 再对其 permute 进入 attention 计算

但需要注意：此时 k 是 **4D tensor**（含 `ns` 序列维度），若 `ns > 1`，需要按序列分别处理。

## 5. Phase 计划

### Phase 1: GGML 张量层定义
- `ggml.h`: 新增 `GGML_OP_KMEANS` 枚举
- `ggml.h`: 声明 `ggml_kmeans()` API
- `ggml.h` / 对应实现: 注册 `ggml_op_name()` 描述

### Phase 2: CPU 后端实现（ARM NEON）
- `ops.h`: 声明 `ggml_compute_forward_kmeans()`
- `ops.cpp`: 实现核心 KMeans（F32 计算路径）
  - dequantize F16/BF16 → F32
  - 均匀采样初始化
  - 迭代：assignment + centroid update
  - 按最小 token 索引排序并重编号 cluster
  - sink token 标记为 -1
- `ggml-cpu.c`: switch-case 注册 `GGML_OP_KMEANS`
- ARM NEON 优化 cosine distance 批量计算

### Phase 3: 后端能力声明
- `ggml-cpu.cpp`: `supports_op()` 中声明支持

### Phase 4: 模型集成
- `src/llama-graph.cpp`: 添加 `build_kmeans()` 辅助函数
- `src/models/llama.cpp`: 在 attention 图构建中插入算子

### Phase 5: 测试
- `tests/test-backend-ops.cpp`: 添加 `test_kmeans` 测试用例
- Android ARM64 交叉编译验证
- 数值正确性 + 性能基准


---

## 6. 实现方案附录

### 6.1 文件修改清单 (实际执行)

| 文件 | 操作 |
|------|------|
| `ggml/include/ggml.h:577` | `GGML_OP_KMEANS` 枚举 |
| `ggml/include/ggml.h:2409` | `ggml_kmeans()` API 声明 |
| `ggml/src/ggml.c:5360` | `ggml_kmeans()` 构造函数 |
| `ggml/src/ggml-cpu/ops.h:115` | `ggml_compute_forward_kmeans` 声明 |
| `ggml/src/ggml-cpu/vec.h:23` | `GGML_KMEANS_CHUNK_SIZE` / `GGML_KMEANS_MAX_CLUSTERS` 常量 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | 三处 switch case: forward dispatch / n_tasks / work_size |
| `ggml/src/ggml-cpu/ops.cpp` | 核心实现 (约 250 行) |

### 6.2 核心内部 Layout 设计

**centroids_cur** (work buffer, F32): `[d_head * n_clusters]`, cluster-major
```
centroids_cur[c * d_head + d]  =  cluster c, element d
```
用途：存储当前迭代的 centroid 值，用于更新步骤。

**centroids_norm** (work buffer, F32): `[d_head * n_clusters]`, d_head-major (转置)
```
centroids_norm[d * n_clusters + c]  =  L2-normalized centroid c, element d
```
用途：assignment 步骤的高频 cosine similarity 计算。采用 d_head-major 布局，一次加载 4 个连续 cluster 的同一维度元素进入 NEON 寄存器。

**k_f32** (per-head buffer, F32): `[d_head * n_tokens]`
```
k_f32[t * d_head + d]  =  head h, token t, element d
```
整个 head 的 K cache 先整体 dequantize 到 F32，简化后续的随机访问。

**输出 centroids** (F16, 3D tensor): `[d_head, n_heads, n_clusters]`
```
offset(d, h, c) = h * d_head + d + c * d_head * n_heads    (in fp16 units)
```
cluster 维度 stride = `nb[2] / sizeof(ggml_fp16_t)` = `d_head * n_heads`。

### 6.3 ARM NEON 优化策略

核心思路：**标量广播 + 向量 FMA**。

```
对每个 k_vec 元素 d:
  float32x4_t kv_v = vdupq_n_f32(k_vec[d]);            // 广播到 4 lanes
  对每组 4 个 cluster:
    float32x4_t cv = vld1q_f32(&centroids_norm[d*n_clusters + g*4]);
    acc[g] = vfmaq_f32(acc[g], cv, kv_v);              // 4 clusters x 1 k_elem
```

FMA 指令数量分析 (d_head=128, n_clusters=8):
- Scalar broadcast: 128 d_iter x 2 groups = 256 FMAs/token
- 等价于: 128 x 8 = 1024 scalar MUL + 896 scalar ADD = ~1920 ops
- 理论加速: ~7.5x 算术指令减少

为什么不用向量-向量 FMA（4-d_head x 4-cluster 同时计算）：
- 两者 FMA 总数相同 (= d_head x n_clusters / 4)
- Broadcast 版不需要 transpose centroids_norm，节省每次迭代的转置开销
- centroids_norm 按 stride=n_clusters 加载，但 d_head x n_clusters <= 4KB 全部在 L1 中

### 6.4 并行与内存模型

```
并行: ith/nth striping over n_heads
  for h in [ith, ith+nth, ith+2*nth, ...]:
    kmeans_head(h)

单 head 内存 (d_head=128, n_clusters=8, n_tokens=32K):
  临时 k_f32:         d_head * n_tokens * 4 = 16 MB  (per-head, 复用)
  Work buffer:        ~268 KB per thread (线程间独立)
  输出 centroids_f16: d_head * n_clusters * 2 = 2 KB  per head
  输出 assignments:   n_tokens * 4 = 128 KB per head
```

k_f32 的 16MB 是主要内存开销。未来可优化为 chunked dequantize（在 work buffer 中按 CHUNK_SIZE 分批），消除此 buffer。

### 6.5 与设计文档的差异

| 项目 | 设计文档 | 实际实现 | 原因 |
|------|----------|----------|------|
| centroids 类型 | ~~F32~~ F16 | **F16** | 减少显存占用，与 k_cache 类型一致 |
| k_f32 分配 | chunked | **全量** (malloc) | 先简单实现，后续可改为 chunked |
| NEON 策略 | 向量-向量 | **标量广播** | 避免转置开销，FMA 总数相同 |
