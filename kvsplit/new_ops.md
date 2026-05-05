基于代码库分析，我来给你一个**新增算子并集成到模型推理**的完整计划。

---

## 📋 新增 CPU 算子完整计划

### 架构概览

llama.cpp 的算子体系分三层：
1. **GGML 张量层** (`ggml/include/ggml.h`) — 定义算子枚举和构建 API
2. **CPU 后端层** (`ggml/src/ggml-cpu/`) — 实现算子的实际计算（含 ARM NEON SIMD）
3. **模型图构建层** (`src/models/`, `src/llama-graph.cpp`) — 在推理图中调用算子

---

### Phase 1: GGML 张量层定义

**目标**：在 GGML 公共接口中声明新算子。

| 步骤 | 文件 | 操作内容 |
|------|------|----------|
| 1.1 | `ggml/include/ggml.h` | 在 `enum ggml_op` 末尾新增算子枚举，如 `GGML_OP_MY_OP` |
| 1.2 | `ggml/include/ggml.h` | 声明算子构建函数，如 `ggml_my_op(...)`，参数为 `ggml_context*` + 输入张量 + 配置参数 |
| 1.3 | `ggml/include/ggml.h` | 如有需要，声明算子参数获取/设置辅助函数 |

**注意**：
- `ggml_tensor` 的 `op_params[GGML_MAX_OP_PARAMS]`（通常是 `int32_t[18]`）用于存储算子配置参数（如 eps、scale 等），需紧凑编码
- 使用 `ggml_op_params` union 来规范化参数存取
- 算子构建函数内部会调用 `ggml_new_tensor()` 创建输出张量，设置 `.op` 字段和 `op_params`

---

### Phase 2: CPU 后端实现（核心）

**目标**：在 CPU 后端实现算子的前向计算，这是**工作量最大的部分**。

#### 2a. 声明前向函数

| 步骤 | 文件 | 操作内容 |
|------|------|----------|
| 2.1 | `ggml/src/ggml-cpu/ops.h` | 声明 `void ggml_compute_forward_my_op(ggml_compute_params * params, ggml_tensor * dst);` |

#### 2b. 实现算子逻辑

| 步骤 | 文件 | 操作内容 |
|------|------|----------|
| 2.2 | `ggml/src/ggml-cpu/ops.cpp` | 实现 `ggml_compute_forward_my_op` 及其类型特化版本（如 `_f32`、`_f16` 等） |

**实现要点**：
- **并行模式**：参考现有算子，使用 `ith`（线程 id）和 `nth`（总线程数）做 striping 并行：
  ```cpp
  for (int i = ith; i < n_rows; i += nth) {
      // 处理第 i 行/块
  }
  ```
- **数据布局**：`ggml_tensor` 使用 `ne[4]`（元素数）和 `nb[4]`（字节步长），支持任意 strides，不要假设连续
- **类型支持**：通常至少支持 `GGML_TYPE_F32`，根据需求支持 `F16`、`BF16`、量化类型

#### 2c. ARM NEON SIMD 优化（关键）

| 步骤 | 文件 | 操作内容 |
|------|------|----------|
| 2.3 | `ggml/src/ggml-cpu/vec.h` / `vec.cpp` | 如有通用的向量化原语需求，在此添加（如自定义的向量运算、约减等） |
| 2.4 | `ggml/src/ggml-cpu/simd-mappings.h` | 如需新的 SIMD 抽象宏，参考现有 `GGML_F32_VEC_*` / `GGML_F16_VEC_*` 模式添加 |

**ARM NEON 编码规范**：
- 使用 `simd-mappings.h` 中的宏保持跨平台（x86/ARM/SVE）兼容，如：
  ```cpp
  GGML_F32_VEC vx = GGML_F32_VEC_LOAD(x + i);
  GGML_F32_VEC vy = GGML_F32_VEC_LOAD(y + i);
  GGML_F32_VEC vz = GGML_F32_VEC_FMA(vz, vx, vy);  // fused multiply-add
  ```
- 如需直接写 NEON intrinsic，用 `#ifdef __ARM_NEON` 包裹：
  ```cpp
  #ifdef __ARM_NEON
      float32x4_t v0 = vld1q_f32(src + i);
      float32x4_t v1 = vfmaq_f32(acc, v0, scale);
  #endif
  ```
- 对于 FP16，检查 `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` 是否可用，否则用 `vcvt_f32_f16` 转换到 F32 计算
- 参考 `ggml/src/ggml-cpu/arch/arm/quants.c` 学习量化类型的 NEON 处理模式

#### 2d. 注册到调度器

| 步骤 | 文件 | 操作内容 |
|------|------|----------|
| 2.5 | `ggml/src/ggml-cpu/ggml-cpu.c` | 在 `ggml_compute_forward()` 的 `switch(tensor->op)` 中新增 `case GGML_OP_MY_OP:` 分支 |

---

### Phase 3: 后端能力声明

| 步骤 | 文件 | 操作内容 |
|------|------|----------|
| 3.1 | `ggml/src/ggml-cpu/ggml-cpu.cpp` | 在 `ggml_backend_cpu_device_supports_op()` 中声明 CPU 支持该算子（通常直接返回 true，除非有特殊限制） |

---

### Phase 4: 集成到 Llama 模型推理图

**目标**：让 llama（及其他模型）在 forward pass 中实际调用你的算子。

#### 4a. 图构建上下文

| 步骤 | 文件 | 操作内容 |
|------|------|----------|
| 4.1 | `src/llama-graph.h` / `src/llama-graph.cpp` | 如需通用封装（如 `build_my_op()` 辅助函数），在此添加；或者直接在各模型文件中调用 `ggml_my_op()` |

#### 4b. 模型特定集成

| 步骤 | 文件 | 操作内容 |
|------|------|----------|
| 4.2 | `src/models/llama.cpp` | 在 `llm_build_llama` 构造函数中，在合适的层位置插入你的算子调用 |
| 4.3 | 其他模型文件 (`src/models/*.cpp`) | 如需支持更多模型架构，在对应模型的 `llm_build_*` 中插入算子 |

**插入位置示例**（以 Llama 为例）：
```cpp
// 当前在 llama.cpp 的 build 流程中：
cur = build_norm(inpL, model.layers[il].attn_norm, ...);  // RMS Norm
auto [Qcur, Kcur, Vcur] = build_qkv(...);
// ... 在此处或他处插入你的算子
cur = ggml_my_op(ctx0, cur, ...);  // 你的新算子
```

---

### Phase 5: 编译与测试

#### 5a. Android ARM NEON 编译验证

```bash
# 使用 Android NDK 交叉编译
mkdir build-android && cd build-android
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CPU_ARM_ARCH=armv8-a+fp16+dotprod

make -j$(nproc)
```

#### 5b. 功能正确性测试

- 用 `tests/test-backend-ops.cpp` 的模式编写单测，验证输出数值正确性
- 对比参考实现（纯 C 无 SIMD）与 NEON 优化版本的结果一致性
- 测试多种输入 shape、batch size、stride 组合

#### 5c. 性能基准测试

- 使用 `tools/llama-bench/` 对比插入算子前后的推理速度
- 检查 NEON 向量化是否生效（可用 simpleperf 或自打印 cycle count）

---

### 📁 涉及文件清单（按修改顺序）

| 顺序 | 文件路径 | 目的 |
|------|----------|------|
| 1 | `ggml/include/ggml.h` | 算子枚举 + 构建 API |
| 2 | `ggml/src/ggml-cpu/ops.h` | 前向计算函数声明 |
| 3 | `ggml/src/ggml-cpu/ops.cpp` | 算子核心实现（含 NEON SIMD） |
| 4 | `ggml/src/ggml-cpu/vec.h` / `vec.cpp` | 通用向量化原语（如需要） |
| 5 | `ggml/src/ggml-cpu/simd-mappings.h` | SIMD 抽象宏（如需要） |
| 6 | `ggml/src/ggml-cpu/ggml-cpu.c` | 调度器 switch-case 注册 |
| 7 | `ggml/src/ggml-cpu/ggml-cpu.cpp` | 后端能力声明 |
| 8 | `src/models/llama.cpp` | Llama 模型图集成 |
| 9 | `src/models/其他模型.cpp` | 其他模型图集成（如需） |
| 10 | `src/llama-graph.h/cpp` | 通用图构建辅助（可选） |

---

### ⚠️ 关键注意事项

1. **线程安全**：`ops.cpp` 中的实现必须是线程安全的，不同线程处理不同数据切片，不要写共享状态
2. **数据类型**：GGML 支持多种量化类型（Q4_0, Q8_0 等），若你的算子接在量化层之后，需正确处理 `ggml_tensor->type`
3. **内存对齐**：NEON load/store 通常要求 16 字节对齐，使用 `GGML_F32_VEC_LOAD` 宏会自动处理，手动写 intrinsic 时需用 `vld1q_f32`（支持非对齐）或先对齐指针
4. **FP16 兼容**：Android 设备对 FP16 向量算术支持不一，务必写 fallback 路径
5. **图构建时无计算**：`ggml_my_op()` 只是创建图节点，实际计算在 `ggml_graph_compute()` 时发生
6. **Batch 维度**：LLM 推理有 prompt processing（大 batch）和 token generation（batch=1）两种模式，算子实现要高效处理这两种情况

---

### 🎯 后续扩展（Hexagon 后端）

等你后续要移植到 Hexagon 后端时，需要：
1. 在 `ggml/src/ggml-hexagon/` 下实现对应的 `ggml_compute_forward_my_op`
2. 在 Hexagon 后端注册 `supports_op`
3. 图构建代码**无需修改**（因为图是 backend-agnostic 的）

---

如果你愿意告诉我**具体是什么类型的算子**（自定义激活函数？注意力变体？量化/反量化？融合算子？），我可以进一步帮你细化每个文件的具体修改方案和代码模板。