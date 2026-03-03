# LightGlue 到 MambaGlue 迁移总结

## 修改完成的文件

### 1. train_onPureGen_v2.py ✅
**用途**: 使用生成数据训练 MambaGlue

**主要修改**:
- 导入: `from lightglue import LightGlue, SuperPoint` → `from mambaglue import MambaGlue, SuperPoint`
- 类名: `PL_LightGlue_Gen` → `PL_MambaGlue_Gen`
- 匹配器: `LightGlue(**lg_conf)` → `MambaGlue(**mg_conf)`
- 添加 SuperPoint 预训练权重加载 (URL: https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth)
- 结果路径: `results/lightglue_gen/` → `results/mambaglue_gen/`
- 默认名称: `lightglue_gen_baseline` → `mambaglue_gen_baseline`

**数据集**:
- 训练集: `data/260227_2_v29_2_1` (生成数据)
- 验证集: `data/CFFA` (真实数据)

---

### 2. train_onReal.py ✅
**用途**: 使用真实数据训练 MambaGlue

**主要修改**:
- 导入: `from lightglue import LightGlue, SuperPoint` → `from mambaglue import MambaGlue, SuperPoint`
- 类名: `PL_LightGlue_Real` → `PL_MambaGlue_Real`
- 匹配器: `LightGlue(**lg_conf)` → `MambaGlue(**mg_conf)`
- 添加 SuperPoint 预训练权重加载
- 结果路径: `results/lightglue_cffa/` → `results/mambaglue_cffa/`
- 默认名称: `lightglue_cffa_baseline` → `mambaglue_cffa_baseline`

**数据集**:
- 训练集: `data/operation_pre_filtered_cffa` (真实数据)
- 验证集: `data/CFFA` (真实数据)

---

### 3. test_onPureGen_v2.py ✅
**用途**: 在生成数据上测试 MambaGlue

**主要修改**:
- 导入: `from lightglue import LightGlue, SuperPoint` → `from mambaglue import MambaGlue, SuperPoint`
- 类名: `PL_LightGlue_Gen` → `PL_MambaGlue_Gen`
- 匹配器: `LightGlue(**lg_conf)` → `MambaGlue(**mg_conf)`
- 添加 SuperPoint 预训练权重加载
- 结果路径: `results/lightglue_gen/` → `results/mambaglue_gen/`
- 描述: "LightGlue Gen-Data Testing" → "MambaGlue Gen-Data Testing"

**数据集**:
- 测试集: `data/CFFA` (真实数据验证集)

---

### 4. test_onReal.py ✅
**用途**: 在真实数据上测试 MambaGlue

**主要修改**:
- 导入: `from lightglue import LightGlue, SuperPoint` → `from mambaglue import MambaGlue, SuperPoint`
- 类名: `PL_LightGlue_Real` → `PL_MambaGlue_Real`
- 匹配器: `LightGlue(**lg_conf)` → `MambaGlue(**mg_conf)`
- 添加 SuperPoint 预训练权重加载
- 结果路径: `results/lightglue_cffa/` → `results/mambaglue_cffa/`
- 描述: "LightGlue CFFA Real-Data Testing" → "MambaGlue CFFA Real-Data Testing"

**数据集**:
- 测试集: `data/CFFA` (真实数据验证集)

---

## 关键改进

### 1. SuperPoint 预训练权重
所有脚本现在都会自动下载并加载 SuperPoint 预训练权重：
```python
sp_url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"
try:
    sp_state = torch.hub.load_state_dict_from_url(sp_url, map_location='cpu')
    self.extractor.load_state_dict(sp_state, strict=False)
    logger.info("成功加载 SuperPoint 预训练权重")
except Exception as e:
    logger.warning(f"加载 SuperPoint 预训练权重失败: {e}，使用随机初始化")
```

### 2. 数据集路径统一
所有数据集路径已更新为相对于项目根目录的 `data/` 文件夹：
- `data/260227_2_v29_2_1/` - 生成数据
- `data/CFFA/` - CFFA 真实数据（验证/测试）
- `data/operation_pre_filtered_cffa/` - 预过滤的 CFFA 真实数据（训练）

### 3. 验证逻辑统一
训练脚本的验证逻辑现在直接调用测试脚本中的前向逻辑和指标计算：
- `train_onPureGen_v2.py` 验证时使用与 `test_onPureGen_v2.py` 相同的逻辑
- `train_onReal.py` 验证时使用与 `test_onReal.py` 相同的逻辑
- 这确保了训练验证指标与测试指标的一致性

---

## 使用示例

### 训练 MambaGlue (生成数据)
```bash
python scripts/v1/train_onPureGen_v2.py \
    --name mambaglue_gen_exp1 \
    --batch_size 4 \
    --num_workers 8 \
    --max_epochs 200 \
    --gpus 0
```

### 训练 MambaGlue (真实数据)
```bash
python scripts/v1/train_onReal.py \
    --name mambaglue_cffa_exp1 \
    --batch_size 4 \
    --num_workers 8 \
    --max_epochs 200 \
    --gpus 0
```

### 测试 MambaGlue (生成数据)
```bash
python scripts/v1/test_onPureGen_v2.py \
    --name mambaglue_gen_exp1 \
    --test_name test_results \
    --batch_size 4 \
    --gpus 0
```

### 测试 MambaGlue (真实数据)
```bash
python scripts/v1/test_onReal.py \
    --name mambaglue_cffa_exp1 \
    --test_name test_results \
    --batch_size 4 \
    --gpus 0
```

---

## 注意事项

1. **依赖检查**: 确保已安装 `mambaglue` 模块及其依赖（mamba_ssm 等）
2. **数据集准备**: 确保所有数据集已正确放置在 `data/` 目录下
3. **GPU 内存**: MambaGlue 可能比 LightGlue 需要更多 GPU 内存，根据需要调整 batch_size
4. **预训练权重**: 首次运行时会自动下载 SuperPoint 权重，需要网络连接
5. **结果目录**: 训练和测试结果会保存在 `results/mambaglue_*/` 目录下

---

## 配置参数

所有脚本使用相同的配置结构：
```python
conf.MATCHING = {
    'features': 'superpoint',
    'input_dim': 256,
    'descriptor_dim': 256,
    'depth_confidence': -1,  # 训练时禁用早停
    'width_confidence': -1,
    'filter_threshold': 0.1,
    'flash': False
}
```

根据 MambaGlue 的具体实现，可能需要调整这些参数。

---

## 迁移完成时间
2025-03-02

## 修改者
AI Assistant (Claude)
