# KV Cache Eviction 算子设计文档

## 1. 背景与目标

在 LLM 推理的 prefill 阶段，KV cache 随序列长度线性增长，长上下文场景下显存压力大。通过对 prefill 产生的 KV cache 做 per-head 稀疏激活——每个 head 只保留最重要的 cluster 所包含的 token——可以大幅压缩 KV cache，降低后续 attention 的计算量。

### 与 KMeans 算子的关系

本算子依赖 `ggml_kmeans` 的聚类结果：

```
KMeans 输出:
  centroids    [d_head, n_heads, n_clusters]  F16
  assignments  [n_tokens, n_heads]            I32
```

在此基础上，本算子完成两个新功能：
1. **投票**：用 prefill 末尾若干 token 的 Q 向量，评估每个 cluster 的重要性
2. **裁剪 + 重排**：保留重要 cluster 的成员 token（受 memory budget 限制），丢弃其余，按 **head-major 物理布局** 重新打包

### 核心假设

- 只处理 **prefill 阶段**的 KV cache。Decoding 阶段 token 少且逐 token 生成，不使用此压缩路径。
- prefill 和 decoding 使用**独立的 KV cache 存储方式**，物理布局不同：
  - 正常 KV cache（sink + decoding）：**token-major** 物理布局
  - Evicted KV cache（prefill 保留）：**head-major** 物理布局
- **K cache 与 V cache 严格同步**：同一 token 要么 K/V 同时保留，要么同时丢弃
- 每 head 独立决策，但输出受 `memory_budget` 硬性上限约束

---

## 2. 算子语义

### 2.1 输入

| 输入 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `k_cache` | `[d_head, n_heads, n_tokens]` | F16 | 原始 K cache（prefill 产生，**token-major 物理布局**） |
| `v_cache` | `[d_head, n_heads, n_tokens]` | F16 | 原始 V cache（**必须与 k_cache 同步裁剪**） |
| `q_obs` | `[d_head, n_heads, obs_win]` | F16 | prefill 末尾 `obs_win` 个 token 的 Q 向量 |
| `centroids` | `[d_head, n_heads, n_clusters]` | F16 | KMeans 聚类输出的 centroid |
| `assignments` | `[n_tokens, n_heads]` | I32 | KMeans 聚类输出的每 token cluster 分配 |

> **K/V 同步约束**：`k_cache` 与 `v_cache` 必须是同一组 token 的 K/V 表示。裁剪时，`assignments[t, h]` 同时决定 K 和 V 中 token `t` 的去留。

### 2.2 参数（op_params 编码）

| 参数 | op_params 索引 | 类型 | 说明 |
|------|---------------|------|------|
| `n_clusters` | `[0]` | int32 | 每 head 的 cluster 总数（与 KMeans 一致） |
| `obs_win` | `[1]` | int32 | 投票使用的观察窗口大小（末尾 token 数） |
| `memory_budget` | `[2]` | int32 | 每 head 最多保留的 token 数（硬性上限） |

### 2.3 输出

| 输出 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `dst` (k_evicted) | `[d_head, memory_budget, n_heads]` | F16 | 裁剪后的 K cache（**head-major 物理布局**） |
| `src[2]` (v_evicted) | `[d_head, memory_budget, n_heads]` | F16 | 裁剪后的 V cache（**head-major 物理布局**，与 K 同步） |

> **memory_budget 截断机制**：输出 tensor 的 shape 固定为 `memory_budget`，不因 head 而异。若某 head 实际保留 token 数 `n_kept_h < memory_budget`，则 `memory_budget - n_kept_h` 的位置为 padding（未初始化，attention 时通过 mask 忽略）。
>
> 若 `n_kept_h > memory_budget`（理论上不应发生，因 budget 为硬性截断），compute_forward 中触发 `GGML_ABORT`。

### 2.4 投票逻辑

对每个 head `h`，用 `q_obs[:, h, :]`（即 `obs_win` 个查询向量）投票给 `n_clusters` 个 centroid：

```
for c in [0, n_clusters):
    score[c] = 0
    for t in [0, obs_win):
        score[c] += dot_product(q_obs[:, h, t], centroids[:, h, c])
    score[c] /= obs_win   // 取均值，避免 obs_win 大小影响选票
```

思想：`dot(Q, centroid)` 反映"最近的 token 想通过 attention 关注这个 cluster 的程度"。取均值后，得分高的 cluster 是"被最近 token 最想关注的 cluster"。

### 2.5 裁剪规则（Memory Budget 截断）

1. 对每个 head 的 `n_clusters` 个 score 从大到小排序
2. 按 score 从高到低依次遍历 cluster：
   - 将该 cluster 的所有成员 token 标记为"保留"
   - 累计保留 token 数 `n_kept`
   - **若 `n_kept >= memory_budget`，立即停止，当前 cluster 超出的 token 被截断丢弃**
3. 其余 token 直接丢弃

> **截断示例**：budget = 100，当前已保留 80 个 token，下一个 cluster 有 30 个 token → 只保留前 20 个，后 10 个丢弃。
>
> **sink token 处理**：`assignments[t, h] == -1` 的 sink token **不参与聚类**，在裁剪遍历中自动跳过（即 sink token 不进入 evicted buffer，留在正常 KV cache 中）。

---

## 3. 物理布局与双 KV Cache 设计

### 3.1 输入布局（token-major，标准 GGML 3D）

```
k_cache [d_head, n_heads, n_tokens]:
  k_cache[d, h, t] = offset d*nb[0] + h*nb[1] + t*nb[2]
  nb[0] = sizeof(fp16)
  nb[1] = d_head * sizeof(fp16)
  nb[2] = d_head * n_heads * sizeof(fp16)

  连续读取同一 token 的 d_head 个元素：连续
  连续读取同一 head 的所有 token：stride = nb[2]/2 = d_head * n_heads（不连续！）
```

**关键特性**：同一 head 的相邻 token 在物理内存中相隔 `d_head * n_heads * sizeof(fp16)`。

### 3.2 输出布局（head-major，物理连续）

```
k_evicted [d_head, memory_budget, n_heads]:
  k_evicted[d, tok, h] = offset d*nb[0] + tok*nb[1] + h*nb[2]
  nb[0] = sizeof(fp16)
  nb[1] = d_head * sizeof(fp16)
  nb[2] = d_head * memory_budget * sizeof(fp16)

  连续读取同一 head 的所有保留 token：连续（tok 维度 stride = d_head * sizeof(fp16)）
  不同 head 的数据物理隔离（间隔 d_head * memory_budget * sizeof(fp16)）
```

**关键特性**：同一 head 的相邻 token 在物理内存中**连续**。

### 3.3 为什么需要两种物理布局

| | 正常 KV cache（token-major） | Evicted KV cache（head-major） |
|---|---|---|
| **物理布局** | token 维度连续 | head 维度连续 |
| **同一 head 相邻 token** | **不连续**（相隔 n_heads） | **连续** |
| **裁剪后** | 保留 token 与原 token 交错，无法物理分离 | 每 head 保留 token 物理聚合 |
| **后续 attention 读取** |  stride 大，缓存不友好 |  stride 小，缓存友好 |

**为什么不能直接 concat**：
- 正常 KV cache 是 token-major，evicted KV cache 是 head-major
- `ggml_concat` 要求 concat 维度上其他维度的 stride 一致
- 两者的 `nb[1]` 和 `nb[2]` 完全不同，无法直接 concat
- 必须通过 **combined attention** 分别读取两个 source，然后合并 softmax

### 3.4 双 KV Cache 架构

```
┌──────────────────────────────────────────────────────────────┐
│  正常 KV cache (token-major 物理布局)                         │
│  包含: sink tokens + decoding tokens                         │
│  shape: [d_head, n_head_kv, n_sink + n_dec, ns]             │
│  来源: llama_kv_cache 标准存储                               │
│  不变, 每步 decoding 追加 1 个 token                         │
├──────────────────────────────────────────────────────────────┤
│  Evicted KV cache (head-major 物理布局)                       │
│  包含: prefill 中被保留的 token (按 memory_budget 截断)       │
│  shape: [d_head, memory_budget, n_heads]                     │
│  来源: ggml_kv_evict 输出                                    │
│  prefill 结束后冻结不变                                      │
└──────────────────────────────────────────────────────────────┘
```

**Attention 计算**：
```
attn = combined_softmax( Q @ K_normal^T / √d ,  Q @ K_evicted^T / √d ) @ [V_normal | V_evicted]
```

由于两个 K/V source 物理布局不同，必须用 `GGML_OP_KV_ATTN_COMBINED` 算子内部分别做 matmul，然后合并 softmax。

---

## 4. GGML API 设计

```cpp
// KV Cache eviction: 基于 Q 对 centroid 投票，按 memory_budget 保留 token
// 输出 K/V 为 head-major 物理布局，与正常 KV cache 的 token-major 不同
//
// 参数通过 op_params 编码: [n_clusters, obs_win, memory_budget]
//
// k_cache:      [d_head, n_heads, n_tokens],    type F16,  token-major
// v_cache:      [d_head, n_heads, n_tokens],    type F16,  token-major (与 K 同步)
// q_obs:        [d_head, n_heads, obs_win],     type F16
// centroids:    [d_head, n_heads, n_clusters],  type F16 (KMeans 输出)
// assignments:  [n_tokens, n_heads],             type I32 (KMeans 输出)
//
// 返回 k_evicted: [d_head, memory_budget, n_heads], type F16 (dst, head-major)
// 附带 v_evicted (src[2], head-major, 与 K 同步)
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

> **重要**：`v_cache` 必须与 `k_cache` 是同一组 token 的 V 表示。compute_forward 中 `k_cache` 和 `v_cache` 的同一 token 同时保留或同时丢弃。

---

## 5. 完整 Attention 计算方案：双路 KV + 合并 Attention

### 5.1 核心约束

sink 和 decoding 的 KV cache **必须保持标准格式不动**。原因是：
- decoding 阶段每步追加 1 个新 token，需要直接往 KV cache 末尾写，格式一变就没法正常拼接
- 因果 mask、跨序列 mask 等现有逻辑全部依赖标准 KV cache 结构

Evicted KV cache 使用**独立的 head-major 物理布局**，无法与正常 KV cache 直接 concat。

因此正确的做法是**两种 KV 并存**，通过 combined attention 算子分别查询后合并：

```
┌──────────────────────────────────────────────────────────┐
│  正常 KV cache (token-major 物理布局)                     │
│  包含: sink tokens + decoding tokens                      │
│  shape: [d_head, n_head_kv, n_sink + n_dec, ns]        │
│  不变, 不压缩                                              │
├──────────────────────────────────────────────────────────┤
│  Evicted KV cache (head-major 物理布局)                   │
│  包含: prefill 中被保留的 token (memory_budget 截断)      │
│  shape: [d_head, memory_budget, n_heads]                 │
│  物理布局与正常 KV 不同，无法 concat                       │
└──────────────────────────────────────────────────────────┘
```

### 5.2 合并 Attention 算子 `ggml_kv_attn_combined`

由于两个 K/V source 物理布局不同，必须封装为新算子：

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
// kept_count:   [n_heads]                               (每 head 实际保留的 token 数, 用于 ragged mask)
//
// 参数 op_params: [kq_scale (float)]
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

**为什么要 kept_count 而不是 kept_offsets**：
- 输出 `k_evict` 的 shape 固定为 `memory_budget`，每 head 的保留 token 从 `offset = h * d_head * memory_budget` 开始
- `kept_count[h]` 标定该 head 实际有效的 token 数（`0` 到 `memory_budget`）
- attention 内部对 `[kept_count[h], memory_budget)` 的 padding 部分忽略

### 5.3 compute_forward 流程

```
ggml_compute_forward_kv_attn_combined:

  for each head h (parallel ith/nth):

    n_kept_h = kept_count[h]          // 该 head 实际保留的 token 数 (<= memory_budget)
    n_norm   = n_kv_normal

    // --- K_evict_h 定位 ---
    // k_evict 布局: [d_head, memory_budget, n_heads], head-major
    // head h 的数据在物理内存中连续: offset = h * d_head * memory_budget (in fp16 elements)
    k_evict_h = (fp16*)k_evict->data + h * d_head * memory_budget
    v_evict_h = (fp16*)v_evict->data + h * d_head * memory_budget
    // 有效范围: [0, n_kept_h), padding: [n_kept_h, memory_budget)

    // --- Part A: 正常 KV attention ---
    // Q_h: [d_head, n_tokens], K_normal_h: [d_head, n_norm]
    scores_normal = Q_h^T @ K_normal_h / scale          // [n_tokens, n_norm]
    scores_normal += mask_normal_h                       // 因果/跨序列/空cell mask

    max_normal = max(scores_normal, axis=1)
    sumexp_normal = sum(exp(scores_normal - max_normal), axis=1)

    // --- Part B: Evicted attention ---
    // K_evict_h: [d_head, n_kept_h]  (head-major 物理连续)
    scores_evict = Q_h^T @ K_evict_h / scale             // [n_tokens, n_kept_h]
    // evicted tokens 无需因果 mask（均为历史 token）

    max_evict = max(scores_evict, axis=1)
    sumexp_evict = sum(exp(scores_evict - max_evict), axis=1)

    // --- Part C: 联合 softmax + 加权合并 ---
    max_total = max(max_normal, max_evict)
    sumexp_total = sumexp_normal * exp(max_normal - max_total)
                 + sumexp_evict  * exp(max_evict  - max_total)

    // V_normal_h: [d_head, n_norm], V_evict_h: [d_head, n_kept_h]
    output_h = softmax(scores_normal, max_total, sumexp_total) @ V_normal_h
             + softmax(scores_evict,  max_total, sumexp_total) @ V_evict_h

    写入 dst[:, h, :]
```

### 5.4 为什么不用 flash attention

合并 attention 涉及两个独立 K/V source，现有 `ggml_flash_attn_ext` 不支持多 K/V 输入。第一版用 vanilla attention 分解实现，正确性优先。后续若成为热点，可以写一个融合 kernel。

### 5.5 完整 prefill 流水线

```
每个 transformer 层:
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  1. build_qkv                                               │
  │     Qcur, Kcur, Vcur = Wqkv × input                        │
  │                                                             │
  │  2. 存入标准 KV cache (sink + decoding 路径, 不变)           │
  │     cpy_k(k_cache, Kcur, k_idxs)  // token-major            │
  │     cpy_v(v_cache, Vcur, v_idxs)  // token-major            │
  │                                                             │
  │  3. KMeans 聚类 (对 prefill 的全部 K, sink 之后的 token)     │
  │     centroids, assignments = ggml_kmeans(k_full, ...)       │
  │                                                             │
  │  4. KV Eviction (投票 + 按 memory_budget 截断 + head-major) │
  │     k_evict, v_evict = ggml_kv_evict(k_full, v_full,       │
  │                                     Q_obs, centroids,       │
  │                                     assignments, ...)       │
  │     // k_evict/v_evict 物理布局为 head-major                │
  │     // 与正常 KV cache (token-major) 物理结构不同            │
  │                                                             │
  │  5. 合并 Attention                                          │
  │     K_normal = mctx_cur->get_k(ctx0, il)   // token-major   │
  │     V_normal = mctx_cur->get_v(ctx0, il)   // token-major   │
  │     output = ggml_kv_attn_combined(ctx0, Qcur,              │
  │                 K_normal, V_normal,                         │
  │                 k_evict, v_evict,                           │
  │                 kq_mask, kept_count, kq_scale)              │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

### 5.6 图中插入点

在 `src/llama-graph.cpp` 的 `build_attn` 函数中，KV cache store 之后：

```cpp
// 现有: KV cache store
ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));

ggml_tensor * k_normal = mctx_cur->get_k(ctx0, il);  // token-major
ggml_tensor * v_normal = mctx_cur->get_v(ctx0, il);  // token-major

if (is_prefill && cparams.kv_evict_enabled
    && k_normal->ne[2] > cparams.kmeans_sink_len + 1) {
    // 3. KMeans
    // 4. Eviction (输出 head-major)
    // 5. Combined Attention (分别读取 token-major 和 head-major)
} else {
    // 原有路径: 标准 attention
    cur = build_attn_mha(q, k_normal, v_normal, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
}
```

---

## 6. Compute Forward 实现概要

### 6.1 `ggml_compute_forward_kv_evict`

并行模型：head 按 `ith/nth` stripe 分配。

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
  float scores[n_clusters];
  memset(scores, 0, n_clusters * sizeof(float));

  for c in [0, n_clusters):
      for t in [0, obs_win):
          dot = dot_product(q_obs[:, h, t], centroids[:, h, c])
          scores[c] += dot;
      scores[c] /= obs_win;

  // === Phase B: 按 score 排序 cluster ===
  cluster_ids = argsort(scores, descending)   // [n_clusters]

  // === Phase C: 按 memory_budget 截断保留 + 同步拷贝 K/V ===
  n_kept = 0;
  for idx in [0, n_clusters):          // 按 score 从高到低遍历 cluster
      c = cluster_ids[idx];
      for t in [0, n_tokens):          // 遍历该 cluster 的所有成员
          if assignments[t, h] == c:
              if n_kept < memory_budget:
                  // 同步拷贝 K 和 V 的同一 token
                  copy k_cache[:, h, t] → k_evicted[:, n_kept, h]
                  copy v_cache[:, h, t] → v_evicted[:, n_kept, h]
                  n_kept++;
              else:
                  break;                // 达到 budget 上限，截断
      if n_kept >= memory_budget:
          break;

  kept_count[h] = n_kept;               // 写入辅助输出
```

**K/V 同步保证**：
- `k_cache[:, h, t]` 和 `v_cache[:, h, t]` 的拷贝操作在同一个 `if n_kept < memory_budget` 分支内
- 同一 token 要么 K/V 同时进入 evicted buffer，要么同时被丢弃

**Work buffer**：
```
per-thread (全部栈分配):
  scores         n_clusters * sizeof(float)        ← 最大 1024*4 = 4KB
  cluster_ids    n_clusters * sizeof(int32_t)      ← 最大 1024*4 = 4KB
  ─────────────────────────────────────────────────
  合计: 最大 ~8KB/thread
```

### 6.2 `ggml_compute_forward_kv_attn_combined`

见 5.3 节。

---

## 7. 文件修改清单

| 阶段 | 文件 | 内容 |
|------|------|------|
| Phase 1 | `ggml/include/ggml.h` | `GGML_OP_KV_EVICT` + `GGML_OP_KV_ATTN_COMBINED` 枚举; 两个 API 声明 |
| Phase 1 | `ggml/src/ggml.c` | 两个算子的构造函数 + OP_NAME/SYMBOL 注册 |
| Phase 2 | `ggml/src/ggml-cpu/ops.h` | 两个 compute_forward 声明 |
| Phase 2 | `ggml/src/ggml-cpu/ops.cpp` | 核心实现 |
| Phase 2 | `ggml/src/ggml-cpu/ggml-cpu.c` | 两个算子的三处 switch 注册 |
| Phase 3 | `src/llama-graph.cpp` | 接入点: KV store 后插入 KMeans → Evict → CombinedAttn |

---

## 8. 边界情况与约束

| 边界 | 处理 |
|------|------|
| `memory_budget >= n_tokens` | 所有 token 保留（无裁剪），但输出仍为 head-major 物理布局 |
| `obs_win > n_tokens` | 断言失败 |
| 某 head 保留 token 数为 0 | Combined Attention 中 `n_kept_h = 0`，退化为纯正常 KV attention |
| `n_kv_normal = 0` (无 sink, 无 dec) | 正常，合并 softmax 退化为纯 evicted attention |
| centroid L2 norm = 0 | 投票 dot product 为 0，不影响排序（所有 score 可能相等） |
| sink token (assignments = -1) | 裁剪遍历时自动跳过，sink token 留在正常 KV cache |
| **K/V 不同步** | 由 compute_forward 保证：同一 token 的 K/V 同时拷贝或同时丢弃 |

---

## 9. 与后续 decoding 阶段的衔接

prefill 结束后：
- **正常 KV cache** 继续用于 decoding，每步追加 1 个 token，token-major 格式不变
- **evicted KV cache** 冻结不变（prefill 的压缩结果），head-major 格式不变
- Combined attention 在 decoding 阶段继续使用：`K_normal` 随 decoding 步数增长（token-major），`K_evict` 保持不变（head-major）

后续优化方向：
- 若 decoding 阶段 token 数较多（长输出），可对 decoding cache 也周期性触发压缩
- Combined attention 可进一步优化为分别针对 token-major 和 head-major 的融合 kernel
