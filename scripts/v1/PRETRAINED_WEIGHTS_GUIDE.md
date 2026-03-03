# MambaGlue 预训练权重使用指南

## 当前状态

### SuperPoint（特征提取器）✅
- **状态**: 已配置自动加载预训练权重
- **权重来源**: `https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth`
- **训练状态**: 冻结（不参与训练）
- **位置**: 所有训练和测试脚本中

### MambaGlue（匹配器）⚠️
- **状态**: 目前从零开始训练
- **原因**: MambaGlue 类中的预训练权重加载需要本地文件 `checkpoint_best.tar`
- **训练状态**: 可训练

---

## 问题分析

在 `mambaglue/mambaglue.py` 中，MambaGlue 的 `__init__` 方法会尝试加载权重：

```python
# 当前代码（第 600 行左右）
local_path = Path("checkpoint_best.tar")
if not local_path.exists():
    raise FileNotFoundError(
        f"Weights file not found at {local_path}. Please download it manually."
    )
```

这会导致：
1. 如果没有 `checkpoint_best.tar` 文件，初始化 MambaGlue 时会报错
2. 无法从零开始训练

---

## 解决方案

### 方案 1：禁用自动加载，从零开始训练（推荐用于训练）

修改训练脚本中的 MambaGlue 初始化部分：

**文件**: `train_onPureGen_v2.py`, `train_onReal.py`

**修改位置**: 在 `PL_MambaGlue_*` 类的 `__init__` 方法中

```python
# 原代码
mg_conf = config.MATCHING.copy()
self.matcher = MambaGlue(**mg_conf)

# 修改为
mg_conf = config.MATCHING.copy()
mg_conf['features'] = None  # 禁用自动加载，从零开始训练
self.matcher = MambaGlue(**mg_conf)
```

**优点**:
- 可以从零开始训练 MambaGlue
- 不需要预训练权重文件
- 适合在你的数据集上完全训练

**缺点**:
- 训练时间更长
- 可能需要更多数据才能收敛

---

### 方案 2：使用 MambaGlue 预训练权重（如果有）

如果你有 MambaGlue 的预训练权重文件（例如从 LightGlue 转换或其他来源）：

#### 步骤 1: 准备权重文件

将权重文件放在项目根目录，命名为 `checkpoint_best.tar`，或者修改路径。

#### 步骤 2: 修改训练脚本

```python
# 在 PL_MambaGlue_* 类的 __init__ 方法中
mg_conf = config.MATCHING.copy()
# 保持 features='superpoint'，这样会尝试加载权重
self.matcher = MambaGlue(features='superpoint', **mg_conf)
```

#### 步骤 3: 确保权重文件格式正确

权重文件应该是一个包含 `'model'` 键的字典：
```python
{
    'model': {
        'transformermambas.0.weight': ...,
        'log_assignment.0.weight': ...,
        ...
    }
}
```

---

### 方案 3：手动加载预训练权重（灵活）

在训练脚本中添加手动加载逻辑：

```python
# 在 PL_MambaGlue_* 类的 __init__ 方法中
mg_conf = config.MATCHING.copy()
mg_conf['features'] = None  # 先禁用自动加载
self.matcher = MambaGlue(**mg_conf)

# 手动加载预训练权重（如果存在）
mambaglue_ckpt_path = "path/to/your/mambaglue_pretrained.pth"
if os.path.exists(mambaglue_ckpt_path):
    try:
        mg_state = torch.load(mambaglue_ckpt_path, map_location='cpu')
        # 如果权重文件包含 'model' 键
        if 'model' in mg_state:
            mg_state = mg_state['model']
        # 加载权重（strict=False 允许部分加载）
        self.matcher.load_state_dict(mg_state, strict=False)
        logger.info(f"✅ 成功加载 MambaGlue 预训练权重: {mambaglue_ckpt_path}")
    except Exception as e:
        logger.warning(f"⚠️ 加载 MambaGlue 预训练权重失败: {e}，从零开始训练")
else:
    logger.info("ℹ️ 未找到 MambaGlue 预训练权重，从零开始训练")
```

---

## 推荐的修改步骤

### 立即修改（避免报错）

修改 `mambaglue/mambaglue.py` 中的权重加载逻辑，使其不报错：

**位置**: 第 600 行左右

```python
# 原代码
local_path = Path("checkpoint_best.tar")
if not local_path.exists():
    raise FileNotFoundError(
        f"Weights file not found at {local_path}. Please download it manually."
    )

# 修改为
local_path = Path("checkpoint_best.tar")
if not local_path.exists():
    print(f"⚠️ Weights file not found at {local_path}. Training from scratch.")
    state_dict = None  # 不加载权重，从零开始
else:
    checkpoint = torch.load(str(local_path), map_location="cpu")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("✅ Successfully loaded MambaGlue pretrained weights.")
    else:
        raise KeyError(
            "The checkpoint does not contain 'model' key. Available keys are: ",
            checkpoint.keys(),
        )
```

---

## 训练建议

### 从零开始训练 MambaGlue

**优点**:
- 完全适配你的数据集
- 不受预训练数据偏差影响

**建议**:
1. 使用较小的学习率（1e-4）
2. 使用足够的训练数据
3. 监控验证集指标，及时调整
4. 考虑使用课程学习（已在 train_onPureGen_v2.py 中实现）

### 使用预训练权重微调

**优点**:
- 收敛更快
- 需要更少的训练数据
- 初始性能更好

**建议**:
1. 使用更小的学习率（1e-5 到 1e-4）
2. 可以只微调部分层
3. 使用较少的 epoch

---

## 检查当前配置

运行以下命令检查 MambaGlue 是否能正常初始化：

```python
from mambaglue import MambaGlue

# 测试从零开始
try:
    matcher = MambaGlue(features=None)
    print("✅ MambaGlue 可以从零开始训练")
except Exception as e:
    print(f"❌ 错误: {e}")

# 测试加载权重
try:
    matcher = MambaGlue(features='superpoint')
    print("✅ MambaGlue 成功加载预训练权重")
except Exception as e:
    print(f"⚠️ 无法加载预训练权重: {e}")
```

---

## 总结

**当前状态**:
- ✅ SuperPoint: 使用预训练权重（已配置）
- ⚠️ MambaGlue: 需要配置（建议从零开始训练）

**推荐操作**:
1. 修改 `mambaglue/mambaglue.py`，使权重加载不报错
2. 在训练脚本中设置 `features=None`，从零开始训练
3. 如果有预训练权重，使用方案 3 手动加载

**下一步**:
- 开始训练，观察收敛情况
- 如果收敛困难，考虑寻找或转换 LightGlue 权重作为初始化
