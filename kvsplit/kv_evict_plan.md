# KV Cache Eviction 实现计划

> 基于 `kv_evict.md` 设计文档，对照当前代码库现状制定的详细实现计划。
> 本文档不修改任何代码，仅作方案规划。
>
> 前置文档：
> - [`base_kmeans.md`](base_kmeans.md) — KMeans 算子设计（已实现）
> - [`kv_evict.md`](kv_evict.md) — KV Eviction + Combined Attention 设计
> - [`new_ops.md`](new_ops.md) — GGML 新算子添加指南
> - [`test_kmeans.md`](test_kmeans.md) — KMeans 测试文档

---

## 1. 架构概览

需要新增 **2 个 GGML 算子** + **模型图集成** + **参数与控制面**：

```
KMeans (已有)  ────┐
                   ├──→ GGML_OP_KV_EVICT (新) ──→ k_evict, v_evict (head-major)
Q (末尾obs_win) ───┘                                    │
                                                        │
正常KV cache (token-major) ─────────────────────────────┼──→ GGML_OP_KV_ATTN_COMBINED (新) ──→ attention output
                                                        │
                                                        └──→ (k_evict, v_evict, head-major)
```

**关键前置条件（已满足）**：
- `GGML_OP_KMEANS` 已在 `ggml/include/ggml.h:579`，实现于 `ggml/src/ggml-cpu/ops.cpp`
- `GGML_MAX_SRC = 10`（`ggml/include/ggml.h:224`），足够容纳两个新算子的输入/输出
- `GGML_MAX_OP_PARAMS = 64` bytes = 16 个 int32 slot
- `GGML_MAX_DIMS = 4`
- KMeans 输出的 `assignments` shape 为 `[n_tokens, n_heads]` (ne[0]=n_tokens, ne[1]=n_heads)

---

## 2. Phase 1：GGML 张量层定义

### 2.1 新增算子枚举 (`ggml/include/ggml.h`)

在 `GGML_OP_KMEANS` (line 579) 之后、`GGML_OP_COUNT` (line 581) 之前插入：

```c
GGML_OP_KV_EVICT,
GGML_OP_KV_ATTN_COMBINED,
```

### 2.2 `ggml_kv_evict` API 声明 (`ggml/include/ggml.h`)

```cpp
// KV Cache eviction: 基于 Q 对 centroid 投票，按 memory_budget 保留 token
// 输出 K/V 为 head-major 物理布局，与输入的 token-major 不同
// 参数通过 op_params 编码: [n_clusters, obs_win, memory_budget]
//
// k_cache:      [d_head, n_heads, n_tokens],    type F16,  token-major
// v_cache:      [d_head, n_heads, n_tokens],    type F16,  token-major (与 K 同步)
// q_obs:        [d_head, n_heads, obs_win],     type F16
// centroids:    [d_head, n_heads, n_clusters],  type F16 (KMeans 输出)
// assignments:  [n_tokens, n_heads],             type I32 (KMeans 输出)
//
// 返回 k_evicted: [d_head, memory_budget, n_heads], type F16 (dst, head-major)
// 附带 v_evicted (src[5], head-major, 与 K 同步)
GGML_API struct ggml_tensor * ggml_kv_evict(
        struct ggml_context * ctx,
        struct ggml_tensor  * k_cache,
        struct ggml_tensor  * v_cache,
        struct ggml_tensor  * q_obs,
        struct ggml_tensor  * centroids,
        struct ggml_tensor  * assignments,
        int32_t               n_clusters,
        int32_t               obs_win,
        int32_t               memory_budget);
```

**输入输出 src 分配：**

| src 索引 | Tensor | 方向 | Shape | 类型 | 物理布局 |
|----------|--------|------|-------|------|----------|
| src[0] | k_cache | 输入 | `[d_head, n_heads, n_tokens]` | F16 | token-major |
| src[1] | v_cache | 输入 | `[d_head, n_heads, n_tokens]` | F16 | token-major |
| src[2] | q_obs | 输入 | `[d_head, n_heads, obs_win]` | F16 | — |
| src[3] | centroids | 输入 | `[d_head, n_heads, n_clusters]` | F16 | — |
| src[4] | assignments | 输入 | `[n_tokens, n_heads]` | I32 | — |
| src[5] | v_evicted | 输出 | `[d_head, memory_budget, n_heads]` | F16 | **head-major** |
| dst | k_evicted | 输出 | `[d_head, memory_budget, n_heads]` | F16 | **head-major** |

### 2.3 `ggml_kv_attn_combined` API 声明 (`ggml/include/ggml.h`)

```cpp
// Combined attention over two KV sources (normal cache + evicted buffer)
// 两个 source 物理布局不同，无法在 graph 层 concat
//
// q:            [d_head, n_head, n_tokens]
// k_normal:     [d_head, n_head_kv, n_kv_normal, ns]  (标准 KV cache, token-major)
// v_normal:     [d_head, n_head_kv, n_kv_normal, ns]
// k_evict:      [d_head, memory_budget, n_heads]       (压缩后的 evicted buffer, head-major)
// v_evict:      [d_head, memory_budget, n_heads]
// mask_normal:  [n_kv_normal, n_tps, 1, ns]            (现有 kq_mask)
// kept_count:   [n_heads]                               (每 head 实际保留的 token 数)
//
// 返回 output: [d_head, n_tokens, n_head, ns]  (与 build_attn_mha 相同)
GGML_API struct ggml_tensor * ggml_kv_attn_combined(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k_normal,
        struct ggml_tensor  * v_normal,
        struct ggml_tensor  * k_evict,
        struct ggml_tensor  * v_evict,
        struct ggml_tensor  * mask_normal,
        struct ggml_tensor  * kept_count,
        float                 kq_scale);
```

输入共 7 个，`GGML_MAX_SRC = 10` 足够。

### 2.4 构造函数实现 (`ggml/src/ggml.c`)

#### `ggml_kv_evict` 构造函数（~60 行）

参考 `ggml_kmeans()` 的构造函数模式（`ggml/src/ggml.c:5360-5397`）：

1. **参数校验**：
   - `n_clusters > 0`, `obs_win > 0`, `memory_budget > 0`
   - `obs_win <= k_cache->ne[2]`（观察窗口不能超过 token 数）
2. **类型校验**：
   - k_cache, v_cache, q_obs, centroids 均为 `GGML_TYPE_F16`
   - assignments 为 `GGML_TYPE_I32`
3. **Shape 校验**：
   - 各 tensor 的 `ne[0]` (d_head) 和 `ne[1]` (n_heads) 匹配
   - `centroids->ne[2] == n_clusters`
4. **创建输出 tensor**：
   - dst = k_evicted: `ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_head, memory_budget, n_heads)`
   - src[5] = v_evicted: 同上 shape，`ggml_new_tensor_3d`，op = `GGML_OP_NONE`
5. **op_params 编码**：`[0]=n_clusters`, `[1]=obs_win`, `[2]=memory_budget`

#### `ggml_kv_attn_combined` 构造函数（~40 行）

1. **参数校验**：各输入 tensor 非空、shape 兼容
2. **创建输出 tensor**：`ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_head, n_tokens, n_head, ns)`
   - 输出类型为 F32（与标准 attention 一致）
3. **op_params 编码**：使用 `ggml_set_op_params_f32` 存储 kq_scale
4. **src 数组**：src[0]=q, src[1]=k_normal, src[2]=v_normal, src[3]=k_evict, src[4]=v_evict, src[5]=mask_normal, src[6]=kept_count

### 2.5 设计决策：n_max_kept 的确定

**决策：输出 shape 固定为 `memory_budget`**

由于 `memory_budget` 是用户配置的硬性上限（如 512），在构图时已知。输出 tensor 的 shape 直接设为 `memory_budget`，无需运行时动态确定。

- 若某 head 实际保留 token 数 `n_kept_h < memory_budget`：剩余位置为 padding
- 若 `n_kept_h == memory_budget`：恰好填满
- `n_kept_h > memory_budget` 理论上不可能发生（compute_forward 中 budget 截断），但仍保留 `GGML_ASSERT(n_kept <= memory_budget)` 作为保险

**与旧方案（ragged）对比**：

| | 旧方案 (ragged) | 新方案 (memory_budget 矩形) |
|---|---|---|
| 输出 shape | `[d_head, n_max_kept, n_heads]`，n_max_kept 不确定 | `[d_head, memory_budget, n_heads]`，固定 |
| 辅助输出 | kept_offsets [n_heads+1] + kept_count [n_heads] | kept_count [n_heads] 即可 |
| Attention 定位 | 通过 kept_offsets 查表 | 固定 offset = h * d_head * memory_budget |
| padding 浪费 | 无（ragged 无 padding） | 有（memory_budget - n_kept_h） |
| 实现复杂度 | 高（ragged 索引复杂） | 低（固定 stride） |

---

## 3. Phase 2：CPU 后端实现

### 3.1 声明 (`ggml/src/ggml-cpu/ops.h`)

在 `ggml_compute_forward_kmeans` 声明（line 115）之后添加：

```c
void ggml_compute_forward_kv_evict(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_kv_attn_combined(const struct ggml_compute_params * params, struct ggml_tensor * dst);
```

### 3.2 `ggml_compute_forward_kv_evict` 实现 (`ops.cpp`)

**并行模型**：按 head 在 n_threads 间 striping，与 KMeans 一致。

**Per-head 算法流程**：

```
输入：
  dst (k_evicted): [d_head, memory_budget, n_heads]      (head-major)
  src[0] (k_cache): [d_head, n_heads, n_tokens]          (token-major)
  src[1] (v_cache): [d_head, n_heads, n_tokens]          (token-major, 与 K 同步)
  src[2] (q_obs):   [d_head, n_heads, obs_win]
  src[3] (centroids): [d_head, n_heads, n_clusters]
  src[4] (assignments): [n_tokens, n_heads]
  src[5] (v_evicted): [d_head, memory_budget, n_heads]   (head-major, 与 K 同步)
参数 (op_params): n_clusters, obs_win, memory_budget

每个 head h (parallel over heads):
  // === Phase A: 投票 ===
  float scores[n_clusters];  // stack alloc, <= 1024 clusters
  memset(scores, 0, n_clusters * sizeof(float));

  for c in [0, n_clusters):
      for t in [0, obs_win):
          // dot_product(q_obs[:, h, t], centroids[:, h, c])
          float dot = 0;
          for d in [0, d_head):
              dot += q_obs[d, h, t] * centroids[d, h, c];
          scores[c] += dot;
      scores[c] /= obs_win;  // 取均值

  // === Phase B: 按 score 排序 cluster ===
  // argsort scores descending → cluster_rank[c] = rank of cluster c
  int cluster_rank[n_clusters];
  argsort_descending(scores, n_clusters, cluster_rank);

  // === Phase C: 按 memory_budget 截断保留 + 同步拷贝 K/V ===
  // 按 cluster_rank 从高到低遍历，依次保留成员 token，达到 budget 截断
  bool keep_cluster[n_clusters] = {false};
  for c in [0, n_clusters):
      keep_cluster[cluster_rank[c]] = true;  // 标记该 cluster 为保留候选

  n_kept = 0;
  for rank in [0, n_clusters):         // 按得分从高到低
      c = cluster_rank[rank];
      for t in [0, n_tokens):          // 遍历所有 token
          // assignments 布局: ne[0]=n_tokens, ne[1]=n_heads
          int assigned_c = assignments[t, h];
          if assigned_c == c:          // token t 属于当前 cluster
              if n_kept < memory_budget:
                  // 同步拷贝 K 和 V 的同一 token
                  // k_cache 是 token-major: offset = t*d_head*n_heads + h*d_head
                  // k_evicted 是 head-major: offset = h*d_head*memory_budget + n_kept*d_head
                  for d in [0, d_head):
                      k_evicted[d, n_kept, h] = k_cache[d, h, t];
                      v_evicted[d, n_kept, h] = v_cache[d, h, t];
                  n_kept++;
              else:
                  break;                // 达到 budget 上限，截断当前 cluster
      if n_kept >= memory_budget:
          break;

  GGML_ASSERT(n_kept <= memory_budget);
  kept_count[h] = n_kept;
```

**K/V 同步保证**：
- `k_cache[d, h, t]` 和 `v_cache[d, h, t]` 的拷贝操作在同一个 `if n_kept < memory_budget` 分支内
- 同一 token 要么 K/V 同时进入 evicted buffer，要么同时被丢弃

**索引细节**：
- `k_cache` (token-major): `k_cache[d, h, t]` 的物理偏移 = `t * nb[2] + h * nb[1] + d * nb[0]`
- `k_evicted` (head-major): `k_evicted[d, tok, h]` 的物理偏移 = `h * nb[2] + tok * nb[1] + d * nb[0]`
- `nb` 值由 `ggml_tensor->nb[]` 提供，不能假设连续

**Work buffer 需求**：
```
per-thread (全部栈分配，无需 wdata):
  scores         n_clusters * sizeof(float)        ← 最大 1024*4 = 4KB
  cluster_rank   n_clusters * sizeof(int32_t)      ← 最大 1024*4 = 4KB
  keep_cluster   n_clusters * sizeof(bool)         ← 最大 1024*1 = 1KB
  ─────────────────────────────────────────────────
  合计: 最大 ~9KB/thread
```

**类型支持**：第一版只支持 F16 输入（k_cache/v_cache/q_obs/centroids 均为 F16）。BF16 后续扩展。

**NEON 优化**：投票阶段的 dot product 可直接复用现有的 `ggml_vec_dot_f32`（将 F16 dequantize 到 F32 后计算）。投票总计算量 = `obs_win × n_clusters × d_head`，obs_win 默认为 8，即使用到 n_clusters=1024, d_head=128，也仅 ~1M FLOPs/head。NEON 优化优先级低。

### 3.3 `ggml_compute_forward_kv_attn_combined` 实现 (`ops.cpp`)

这是整个方案的**计算核心**，工作量最大。

**并行模型**：按 head 在 n_threads 间 striping。

**Per-head 算法**：

```
输入：
  src[0] (q):          [d_head, n_head, n_tokens]
  src[1] (k_normal):   [d_head, n_head_kv, n_kv_normal, ns]   (token-major)
  src[2] (v_normal):   [d_head, n_head_kv, n_kv_normal, ns]
  src[3] (k_evict):    [d_head, memory_budget, n_heads]       (head-major)
  src[4] (v_evict):    [d_head, memory_budget, n_heads]
  src[5] (mask_normal): [n_kv_normal, n_tps, 1, ns]
  src[6] (kept_count): [n_heads]
参数 (op_params): kq_scale

每个 head h (parallel over heads):
    // GQA head 映射
    h_kv = h * n_head_kv / n_head;

    n_q    = n_tokens;
    n_norm = n_kv_normal;
    n_ev   = kept_count[h];               // 该 head 实际保留的 token 数 (<= memory_budget)

    // 当前 head 在 evicted buffer 中的起始位置
    // k_evict 布局: [d_head, memory_budget, n_heads], head-major
    //   head h 的连续数据在: offset h * d_head * memory_budget (in fp16 elements)
    k_evict_h = (fp16*)((char*)k_evict->data + h * k_evict->nb[2]);
    v_evict_h = (fp16*)((char*)v_evict->data + h * v_evict->nb[2]);
    // 有效范围: [0, n_ev), padding: [n_ev, memory_budget)

    // K_normal_h 的起始位置
    // K_normal 布局: [d_head, n_head_kv, n_kv_normal, ns], token-major
    k_normal_h = (fp16*)((char*)k_normal->data + h_kv * k_normal->nb[1]);
    v_normal_h = (fp16*)((char*)v_normal->data + h_kv * v_normal->nb[1]);
    // 跨 token stride: k_normal->nb[2]

    // === 逐 query token 计算 ===
    // (避免完整 score 矩阵的 O(n_q * n_norm) 内存)
    for each query q_idx in [0, n_q):
        Q_q = Q data for head h, query q_idx: [d_head]

        // --- A: scores_normal (1D, size n_norm) ---
        float scores_norm[n_norm];   // 若 n_norm 太大需 heap alloc
        for j in [0, n_norm):
            // dot_product(Q_q, K_normal_h[:, j])
            scores_norm[j] = dot(Q_q, k_normal_h + j * k_normal->nb[2]/2) * kq_scale;
        // 应用 mask
        for j in [0, n_norm):
            scores_norm[j] += mask_normal[j, q_idx, 0, ns_idx];

        // --- B: scores_evict (1D, size n_ev) ---
        float scores_evict[n_ev];   // 同上，大时 heap alloc
        for k in [0, n_ev):
            // dot_product(Q_q, k_evict_h[:, k])
            // k_evict 是 head-major: k_evict_h[k] 的偏移 = k * k_evict->nb[1]/2
            scores_evict[k] = dot(Q_q, k_evict_h + k * k_evict->nb[1]/2) * kq_scale;
        // evicted tokens 不需要因果 mask（均为历史 token）

        // --- C: Online stable softmax merge ---
        float M_n = max(scores_norm, n_norm);
        float M_e = (n_ev > 0) ? max(scores_evict, n_ev) : -INFINITY;
        float M   = fmaxf(M_n, M_e);

        float sum_n = 0, sum_e = 0;
        for j in [0, n_norm): sum_n += expf(scores_norm[j] - M);
        for k in [0, n_ev):   sum_e += expf(scores_evict[k] - M);
        float inv_total = 1.0f / (sum_n + sum_e);

        // --- D: Weighted V accumulation ---
        float out[d_head] = {0};  // 或 memset

        // V_normal 贡献
        for j in [0, n_norm):
            float w = expf(scores_norm[j] - M) * inv_total;
            for d in [0, d_head):
                out[d] += w * v_normal_h[j * v_normal->nb[2]/2 + d];

        // V_evict 贡献
        for k in [0, n_ev):
            float w = expf(scores_evict[k] - M) * inv_total;
            for d in [0, d_head):
                out[d] += w * v_evict_h[k * v_evict->nb[1]/2 + d];

        // 写入输出
        dst[:, q_idx, h] = out;

输出: dst [d_head, n_tokens, n_head, ns]
```

**关键实现细节**：

1. **双物理布局的 stride 处理**：
   - `k_normal` 是 token-major：跨 token stride = `nb[2]`
   - `k_evict` 是 head-major：跨 token stride = `nb[1]`
   - 必须通过 `ggml_tensor->nb[]` 做步进，不能假设连续

2. **GQA head 映射**：`h_kv = h * n_head_kv / n_head`。当 n_head == n_head_kv 时为恒等映射。

3. **逐 query 性能**：对于 prefill（n_q 大），逐 query token 循环的 matmul 无法利用缓存局部性。第一版接受此性能代价，正确性优先。后续优化：
   - 对 query 做 tile（每次处理 32/64 query），利用 K 缓存的复用
   - 或直接对接 flash attention 的 tiling 框架

4. **exp 数值稳定**：使用 `M = max(M_n, M_e)` 作为偏移，避免 exp 溢出。

**Work buffer 需求**：
```
per-thread (heap alloc, from wdata):
  scores_norm   n_norm * sizeof(float)     ← 主要开销
  scores_evict  memory_budget * sizeof(float)
  out_q         d_head * sizeof(float)     ← 栈分配即可

对于长序列 (n_norm = 32K): ~132KB per thread, 4 threads → ~528KB
```

**解码阶段优化**：n_q = 1 时，逐 query 就是 natural 的，不需要额外优化。

**NEON 优化重点**：
- **Dot product**：可调用 `ggml_vec_dot_f32`（已向量化），或手工展开
- **exp + sum**：可用 `vexpq_f32` (NEON) 加速
- **V 加权累加**：`vfmaq_f32` (NEON FMLA) 做 d_head 维度的乘加

### 3.4 调度器注册 (`ggml/src/ggml-cpu/ggml-cpu.c`)

需要为两个新算子各添加 **3 处 switch case**（参照 KMeans 模式）：

#### (a) Forward dispatch（参照 line 2079）

```c
case GGML_OP_KV_EVICT:
    {
        ggml_compute_forward_kv_evict(params, tensor);
    } break;
case GGML_OP_KV_ATTN_COMBINED:
    {
        ggml_compute_forward_kv_attn_combined(params, tensor);
    } break;
```

#### (b) n_tasks（参照 line 2423）

```c
case GGML_OP_KV_EVICT:
case GGML_OP_KV_ATTN_COMBINED:
    {
        n_tasks = n_threads;
    } break;
```

#### (c) work_size（参照 line 2938）

```c
case GGML_OP_KV_EVICT:
    {
        // 无 wdata 需求，全部栈分配
        cur = 0;
    } break;

case GGML_OP_KV_ATTN_COMBINED:
    {
        const int n_tokens     = (int) node->src[0]->ne[2];    // n_q
        const int n_kv_norm    = (int) node->src[1]->ne[2];    // n_kv_normal
        const int memory_budget = (int) node->src[3]->ne[1];   // memory_budget
        // per-thread: scores_norm + scores_evict
        cur = n_tasks * (n_kv_norm + memory_budget) * sizeof(float);
    } break;
```

### 3.5 后端能力声明 (`ggml/src/ggml-cpu/ggml-cpu.cpp`)

在 `ggml_backend_cpu_device_supports_op()` 中声明支持两个新算子（直接返回 true，无特殊限制）。

---

## 4. Phase 3：参数与控制面

### 4.1 `include/llama.h` — `llama_context_params`

在 `llama_context_params` 中（kmeans 相关字段附近）新增：

```cpp
// KV Eviction configuration
bool    kv_evict_enabled;      // 启用 KV eviction (依赖 KMeans)
int32_t kv_evict_obs_win;      // 投票观察窗口大小 (默认 8)
int32_t kv_evict_budget;       // 每 head 最多保留的 token 数 (默认 512)
```

默认值在 `llama_context_default_params()` 中设置：
```cpp
kv_evict_enabled = false;
kv_evict_obs_win = 8;
kv_evict_budget  = 512;
```

### 4.2 `src/llama-cparams.h` — `llama_cparams`

在 `llama_cparams` 中（kmeans 字段附近）新增：

```cpp
bool    kv_evict_enabled;
int32_t kv_evict_obs_win;
int32_t kv_evict_budget;
```

### 4.3 `common/common.h` — `common_params`

在 `common_params` 中（kmeans 字段附近）新增：

```cpp
// KV Eviction configuration
bool    kv_evict_enabled   = false;
int32_t kv_evict_obs_win   = 8;
int32_t kv_evict_budget    = 512;
```

### 4.4 `common/common.cpp` — 参数传递与 CLI

1. **CLI 参数解析**：
   ```
   --kv-evict              启用 KV eviction
   --kv-evict-obs-win N    投票窗口大小 (默认 8)
   --kv-evict-budget N     每 head 保留 token 上限 (默认 512)
   ```

2. **cparams 传递**（在 `common_context_params_to_llama` 中）：
   ```cpp
   cparams.kv_evict_enabled = params.kv_evict_enabled;
   cparams.kv_evict_obs_win = params.kv_evict_obs_win;
   cparams.kv_evict_budget  = params.kv_evict_budget;
   ```

3. **src/llama-context.cpp** 中 `params → cparams` 拷贝：
   ```cpp
   cparams.kv_evict_enabled = params.kv_evict_enabled;
   cparams.kv_evict_obs_win = params.kv_evict_obs_win;
   cparams.kv_evict_budget  = params.kv_evict_budget;
   ```

---

## 5. Phase 4：模型图集成 (`src/llama-graph.cpp`)

### 5.1 插入位置

在 `build_attn(llm_graph_input_attn_kv *)` 函数中（line ~2171），KV cache store 之后、`build_attn_mha` 调用之前。

**当前代码结构** (lines 2203-2243)：

```cpp
// store to KV cache
ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));

ggml_tensor * q = q_cur;
ggml_tensor * k = mctx_cur->get_k(ctx0, il);   // token-major
ggml_tensor * v = mctx_cur->get_v(ctx0, il);   // token-major

// [KMeans block — 已有代码]
if (cparams.kmeans_enabled && k->ne[2] > cparams.kmeans_sink_len + 1) {
    ...
}

cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
```

**修改后**：

```cpp
// store to KV cache (不变)
ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));

ggml_tensor * q = q_cur;
ggml_tensor * k_normal = mctx_cur->get_k(ctx0, il);   // token-major
ggml_tensor * v_normal = mctx_cur->get_v(ctx0, il);   // token-major

// 判断是否走 Eviction 路径
bool use_eviction = cparams.kmeans_enabled
                 && cparams.kv_evict_enabled
                 && k_normal->ne[2] > cparams.kmeans_sink_len + 1;

if (use_eviction) {
    const int n_clusters = std::max(1, (int)(k_normal->ne[2] / cparams.kmeans_cluster_div));
    const int obs_win    = std::min(cparams.kv_evict_obs_win,
                                    (int32_t)k_normal->ne[2] - cparams.kmeans_sink_len);
    const int budget     = cparams.kv_evict_budget;

    // 1. KMeans clustering
    ggml_tensor * centroids = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16,
                                                  k_normal->ne[0], k_normal->ne[1], n_clusters);
    ggml_set_name(centroids, "kmeans_centroids");
    ggml_tensor * assignments = ggml_kmeans(ctx0, k_normal, centroids,
                                             n_clusters,
                                             cparams.kmeans_max_iter,
                                             cparams.kmeans_sink_len);
    ggml_set_name(assignments, "kmeans_assignments");
    ggml_build_forward_expand(gf, assignments);

    // 2. 提取 Q_obs: prefill 末尾 obs_win 个 token 的 Q
    ggml_tensor * q_obs = ggml_view_3d(ctx0, q_cur,
        q_cur->ne[0], q_cur->ne[1], obs_win,
        q_cur->nb[0], q_cur->nb[1],
        (q_cur->ne[2] - obs_win) * q_cur->nb[2]);
    ggml_set_name(q_obs, "q_obs");

    // 3. KV Eviction (输出 head-major, 与输入 token-major 物理布局不同)
    ggml_tensor * k_evict = ggml_kv_evict(ctx0, k_normal, v_normal, q_obs,
                                           centroids, assignments,
                                           n_clusters, obs_win, budget);
    ggml_set_name(k_evict, "k_evicted");
    ggml_build_forward_expand(gf, k_evict);

    ggml_tensor * v_evict      = k_evict->src[5];   // head-major, 与 K 同步
    ggml_tensor * kept_count   = ...;               // 需要从某处获取，或在 eviction 中创建

    // 4. Combined Attention (分别读取 token-major 和 head-major)
    cur = ggml_kv_attn_combined(ctx0, q, k_normal, v_normal,
                                 k_evict, v_evict,
                                 kq_mask, kept_count, kq_scale);
} else if (cparams.kmeans_enabled && k_normal->ne[2] > cparams.kmeans_sink_len + 1) {
    // KMeans-only path (保持现有代码)
    ...
    cur = build_attn_mha(q, k_normal, v_normal, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
} else {
    // Standard path (不变)
    cur = build_attn_mha(q, k_normal, v_normal, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
}
```

### 5.2 Q_obs 的获取

`q_cur` 在 `build_qkv` 中产生，shape 为 `[d_head, n_head, n_tokens]`。使用 `ggml_view_3d` 切片最后 `obs_win` 列：

```cpp
// offset = (n_tokens - obs_win) * nb[2]，即跳过前面的 token
ggml_tensor * q_obs = ggml_view_3d(ctx0, q_cur,
    q_cur->ne[0],   // d_head
    q_cur->ne[1],   // n_head
    obs_win,        // 只取最后 obs_win 个
    q_cur->nb[0],   // stride unchanged
    q_cur->nb[1],
    (q_cur->ne[2] - obs_win) * q_cur->nb[2]);  // byte offset
```

### 5.3 kept_count 的传递

由于 `ggml_kv_evict` 的输出固定为 `memory_budget`，attention 需要知道每 head 实际有效的 token 数。

**方案**：在 `ggml_kv_evict` 内部创建 `kept_count` tensor（`[n_heads]` I32），作为 `src[6]` 挂载到 dst 上。但 dst 已经用了 src[0..5]。

实际上，dst 的 src 数组可以扩展：
- src[0] = k_cache
- src[1] = v_cache
- src[2] = q_obs
- src[3] = centroids
- src[4] = assignments
- src[5] = v_evicted
- src[6] = kept_count

`GGML_MAX_SRC = 10`，7 个 src 足够。

然后 `ggml_kv_attn_combined` 接收 `kept_count` 作为 src[6]。

### 5.4 kq_mask 的兼容性

Combined Attention 算子内部对 evicted scores **不应用因果 mask**，因为所有 evicted token 在时间上都 ≤ 最后一个 prefill token，对后续 query 都可见。

仅有的 mask 来源是：
- **padding**：通过 `kept_count[h]`（ragged 边界）自然处理，无需显式 mask
- **跨序列隔离**：若 ns > 1，evicted tokens 属于哪个序列需要跟踪。第一版限制 ns=1 可避免此问题

### 5.5 序列维度 (ns) 约束

第一版限制 `ns == 1`（单序列 prefill）。在 `use_eviction` 条件中加入检查：

```cpp
bool is_single_seq = (k_normal->ne[3] == 1);
bool use_eviction = cparams.kmeans_enabled
                 && cparams.kv_evict_enabled
                 && is_single_seq
                 && k_normal->ne[2] > cparams.kmeans_sink_len + 1;
```

多序列 (ns > 1) 的 ragged 交互留待后续。

### 5.6 对 `build_attn(llm_graph_input_attn_kv_iswa *)` 的处理

Sliding window attention 变体（line 2351）也有 KMeans 集成。第一版**跳过 SWA 场景**，仅对标准 KV cache attention 启用 Eviction。

---

## 6. Phase 5：测试

### 6.1 后端算子树测试 (`tests/test-backend-ops.cpp`)

参照 `test_kmeans` 的模式（`tests/test-backend-ops.cpp:7288-7339`）新增两个测试结构。

#### `test_kv_evict` 用例设计

| # | d_head | n_heads | n_tokens | n_clusters | obs_win | memory_budget | 说明 |
|---|--------|---------|----------|------------|---------|---------------|------|
| 1 | 64 | 2 | 64 | 4 | 8 | 32 | 小配置基础验证 |
| 2 | 128 | 4 | 256 | 8 | 8 | 128 | 中等配置 |
| 3 | 64 | 2 | 32 | 4 | 8 | 64 | budget = n_tokens (无裁剪，仅布局转换) |
| 4 | 64 | 2 | 64 | 8 | 8 | 8 | 极端裁剪 (budget 很小) |
| 5 | 32 | 2 | 32 | 4 | 4 | 16 | obs_win 较小 |

验证点：
- k_evicted/v_evicted 中的数据确实来自被保留 cluster 的 token（与 assignments 交叉验证）
- **K/V 同步**：k_evicted[tok, h, d] 和 v_evicted[tok, h, d] 来自同一原始 token
- kept_count[h] <= memory_budget
- sentinel 检测通过（无越界写入）
- bit-exact 两次执行一致（`max_nmse_err() = 0.0`）

测试难点：与 KMeans 不同，KV Evict 依赖 KMeans 的输出（centroids + assignments）作为输入。需要先运行 KMeans 得到中间结果，或手动构造 assignments。

**方案**：在 `build_graph()` 中，先创建虚拟 KMeans 输出（随机初始化 centroids 和 assignments），不做实际 KMeans 运算。这样 KV Evict 可以独立测试。

#### `test_kv_attn_combined` 用例设计

| # | d_head | n_head | n_tokens | n_kv_norm | memory_budget | n_ev_used | 说明 |
|---|--------|--------|----------|-----------|---------------|-----------|------|
| 1 | 32 | 2 | 4 | 8 | 4 | [2, 3] | 基础验证 |
| 2 | 64 | 4 | 8 | 16 | 8 | [4, 6, 2, 5] | ragged heads (不同 n_ev) |
| 3 | 32 | 2 | 4 | 8 | 4 | [0, 0] | n_ev=0 退化 (pure normal attention) |
| 4 | 32 | 2 | 4 | 0 | 8 | [4, 4] | n_kv_norm=0 退化 (pure evicted attention) |

验证点：
- 输出数值与手动 `concat([K_norm, K_evict])` → `softmax(Q @ K^T) @ V` 一致（允许 1e-5 浮点误差）
- ragged head 正确处理（不同 head 使用不同 n_ev）
- degenerate cases (n_ev=0, n_kv_norm=0) 正确

### 6.2 集成测试

编译带有 `--kv-evict` 参数的 `llama-cli`，在真实模型上验证：
- 输出 logits 与不启用 eviction 时的差异在可接受范围
- 显存占用对比（evicted buffer 大小 = 2 × memory_budget × d_head × n_heads × 2 bytes）
- prefill 速度对比

---

## 7. 风险与待解决问题

| 风险 | 严重程度 | 缓解措施 |
|------|----------|----------|
| **Combined Attention 逐 query 实现性能差** | 高 | 第一版接受；后续可 tiling（每次处理 32-64 query）或直接上 flash attention 风格融合 kernel |
| **K/V 不同步** | 高 | compute_forward 中强制同步拷贝：K 和 V 的同一 token 必须在同一分支内同时保留/丢弃 |
| **多序列 (ns > 1) 的 evicted token 归属** | 中 | 第一版限制 ns == 1；预留 ns 扩展点 |
| **GQA head 映射在 combined attention 中的 stride 计算** | 中 | 需要仔细处理 `nb[]` stride，特别是 n_kv_normal < allocated kv_size 时的 padding |
| **KMeans + Evict 增加 prefill 延迟** | 低 | KMeans 已有 perf 基准；Evict 的投票+拷贝计算量远小于 attention matmul |
| **centroid L2 norm = 0 导致投票 dot product 为 0** | 低 | 不影响排序结果，极端情况下所有 score 相等 |
| **sink token 在 eviction 路径中的处理** | 低 | sink token 的 assignments 值为 -1（KMeans 已标记），eviction 遍历时自动跳过 |

---

## 8. 文件修改清单（按阶段顺序）

| 顺序 | 文件 | 内容 | 预估行数 |
|------|------|------|----------|
| 1 | `ggml/include/ggml.h` | 2 个新 OP 枚举 + 2 个 API 声明 | ~50 |
| 2 | `ggml/src/ggml.c` | 2 个构造函数实现 | ~120 |
| 3 | `ggml/src/ggml-cpu/ops.h` | 2 个 forward 函数声明 | ~4 |
| 4 | `ggml/src/ggml-cpu/ops.cpp` | 核心实现 | ~600-800 |
| 5 | `ggml/src/ggml-cpu/ggml-cpu.c` | 3×2 = 6 处 switch case 注册 | ~30 |
| 6 | `ggml/src/ggml-cpu/ggml-cpu.cpp` | supports_op 声明 | ~4 |
| 7 | `include/llama.h` | llama_context_params 新字段 | ~6 |
| 8 | `src/llama-cparams.h` | llama_cparams 新字段 | ~6 |
| 9 | `src/llama-context.cpp` | 默认值 + params→cparams 拷贝 | ~10 |
| 10 | `common/common.h` | common_params 新字段 | ~6 |
| 11 | `common/common.cpp` | CLI 参数解析 + cparams 传递 | ~30 |
| 12 | `common/arg.cpp` | `--kv-evict*` 命令行参数 | ~20 |
| 13 | `src/llama-graph.cpp` | build_attn 中插入 Evict + CombinedAttn | ~50 |
| 14 | `tests/test-backend-ops.cpp` | 2 个测试结构 + 用例注册 | ~200 |

---

## 9. 实现顺序建议

1. **先做 `GGML_OP_KV_EVICT` 的完整链路**（Phase 1-2-3 中 kv_evict 部分 + Phase 5 测试），拿到一个可验证的端到端结果
2. **再做 `GGML_OP_KV_ATTN_COMBINED`**（Phase 1-2-3 中 combined attn 部分 + Phase 5 测试），可独立于模型图验证正确性
3. **最后做 Phase 4 图集成**，将两个算子串入 `build_attn`，配合 Phase 3 参数控制面做端到端验证

---

## 10. 与 `kv_evict.md` 设计文档的差异说明

| 项目 | `kv_evict.md` | 本计划 | 原因 |
|------|---------------|--------|------|
| 核心约束 | `n_keep_clusters` | `memory_budget` | 硬性 token 数上限比 cluster 数更直观，且避免 ragged 的复杂性 |
| 输出 shape | ragged `[d_head, n_max_kept, n_heads]` | 固定 `[d_head, memory_budget, n_heads]` | 固定 shape 简化 attention 索引（offset = h * d_head * memory_budget） |
| 辅助输出 | `kept_offsets[n_heads+1]` + `kept_count[n_heads]` | 仅 `kept_count[n_heads]` | 固定 shape 不需要 offsets 查表 |
| 物理布局 | 文档明确 token-major vs head-major | 一致 | 无差异 |
| K/V 同步 | 文档提及 | 本计划强调 compute_forward 中强制同步拷贝 | 确保同一 token 的 K/V 同时保留/丢弃 |
| 序列维度 | 未提及 | 第一版限制 ns == 1 | 多序列 ragged 交互复杂，先支持核心场景 |
| SWA 支持 | 未提及 | 第一版跳过 | SWA 有独立路径，后续扩展 |
| Combined Attention tiling | 未提及 | 逐 query 计算 | 避免 O(n_q × n_norm) 内存，正确性优先 |
| BF16 支持 | 未提及 | 第一版仅 F16 | 与 KMeans 保持一致，BF16 后续扩展 |
| 解码阶段 | "继续使用 combined attn" | 第一版在 prefill 后冻结 evicted buffer | 与文档一致，按文档 9.1 节执行 |
