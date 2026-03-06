# MambaGlue 视网膜多模态匹配训练 (V3)

本目录包含适配 MambaGlue 的训练和测试代码，用于视网膜多模态图像匹配任务。

## 文件说明

| 文件 | 说明 |
|------|------|
| `gen_data_enhance.py` | 域随机化增强模块，用于训练时数据增强 |
| `train_onMultiGen_vessels_enhanced.py` | 主训练脚本，包含血管引导课程学习 |
| `test_all_operationpre.py` | 测试脚本，在真实数据集上评估模型性能 |

---

## 1. 训练指令

### 基本训练命令

```bash
cd /data/student/Fengjunming/MambaGlue-Retina2603

python scripts/v3/train_onMultiGen_vessels_enhanced.py \
    --name mambaglue_vessel_v3_exp1 \
    --batch_size 4 \
    --num_workers 8 \
    --img_size 512 \
    --max_epochs 200 \
    --gpus 1
```

### 完整参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--name`, `-n` | `mambaglue_vessel_guided_v3` | 实验名称，用于保存结果 |
| `--batch_size` | 4 | 批次大小 |
| `--num_workers` | 8 | 数据加载线程数 |
| `--img_size` | 512 | 输入图像尺寸 |
| `--max_epochs` | 200 | 最大训练轮数 |
| `--gpus` | 1 | GPU设备ID，支持多卡如 `0,1` |
| `--start_point` | None | 从检查点恢复训练 |
| `--teaching_end` | 50 | 课程学习教学期结束epoch |
| `--weaning_end` | 100 | 课程学习断奶期结束epoch |
| `--max_vessel_weight` | 10.0 | 血管损失权重最大值 |
| `--min_vessel_weight` | 1.0 | 血管损失权重最小值 |
| `--patience` | 10 | 早停和学习率调度耐心值 |

### 课程学习说明

训练采用血管引导的课程学习策略：

- **教学期 (Teaching)**: Epoch 0-50，权重=10.0，强迫模型关注血管区域
- **断奶期 (Weaning)**: Epoch 50-100，权重从10.0线性衰减到1.0
- **独立期 (Independence)**: Epoch 100+，权重=1.0，正常模式

### 从检查点恢复训练

```bash
python scripts/v3/train_onMultiGen_vessels_enhanced.py \
    --name mambaglue_vessel_v3_exp1 \
    --start_point results/mambaglue_gen/mambaglue_vessel_v3_exp1/best_checkpoint/model.ckpt \
    --max_epochs 300
```

---

## 2. 测试/评估指令

### 基本测试命令

```bash
python scripts/v3/test_all_operationpre.py \
    --name mambaglue_vessel_v3_exp1 \
    --test_name test_results
```

### 带 Baseline 对比的测试

```bash
python scripts/v3/test_all_operationpre.py \
    --name mambaglue_vessel_v3_exp1 \
    --test_name test_with_baseline \
    --baseline
```

### 完整参数说明

| 参数 | 说明 |
|------|------|
| `--name`, `-n` | 模型名称（对应训练时的 `--name`） |
| `--test_name`, `-t` | 测试名称，用于保存结果目录 |
| `--checkpoint`, `-c` | 检查点路径（默认 `results/mambaglue_gen/<name>/best_checkpoint/model.ckpt`） |
| `--baseline` | 是否运行 MambaGlue 预训练权重作为基准对比 |
| `--batch_size` | 批次大小，默认4 |
| `--num_workers` | 数据加载线程数，默认8 |
| `--gpus` | GPU设备ID |
| `--no_viz` | 禁用可视化输出 |

---

## 3. 训练成果说明

### 训练过程监控指标

训练过程中会监控以下指标：

| 指标 | 说明 |
|------|------|
| `train/loss` | 训练损失（负对数似然损失） |
| `val_loss` | 验证损失 |
| `auc@5` | AUC@5（误差<5像素的匹配比例） |
| `auc@10` | AUC@10（误差<10像素的匹配比例） |
| `auc@20` | AUC@20（误差<20像素的匹配比例） |
| `mAUC` | 平均AUC（对误差分布积分） |
| `combined_auc` | 综合AUC = (AUC@5 + AUC@10 + AUC@20) / 3 |
| `val_mse` | 均方误差（仅统计Acceptable样本） |
| `val_mace` | 角点平均误差（仅统计Acceptable样本） |

### 样本分类标准

根据 `metrics_cau_principle_0304.md`：

- **Failed**: 匹配失败或误差≥1000像素
- **Inaccurate**: 50像素≤MAE<100像素 或 20像素≤MEE<100像素
- **Acceptable**: MAE<50像素 且 MEE<20像素

### 训练结果保存位置

训练结果保存在 `results/mambaglue_gen/<name>/` 目录下：

```
results/mambaglue_gen/<name>/
├── best_checkpoint/      # 最佳模型检查点
│   ├── model.ckpt
│   └── log.txt          # 最佳模型的指标记录
├── latest_checkpoint/    # 最新模型检查点
│   └── model.ckpt
├── metrics.csv           # 所有epoch的指标记录
├── log.txt              # 训练日志
├── epochXX_best/        # 可视化结果（每5个epoch或最佳模型）
│   └── ...
└── epochXX/             # 其他epoch的可视化
```

### TensorBoard 日志

使用 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir logs/tb_logs/mambaglue_<实验名>
```

---

## 4. 测试结果说明

### 测试结果保存位置

测试结果保存在 `results/mambaglue_gen/<name>/<test_name>/` 目录下：

```
results/mambaglue_gen/<name>/<test_name>/
├── test_log.txt              # 测试日志
├── test_summary_trained.txt  # 训练模型测试总结
├── test_summary_baseline.txt # Baseline模型测试总结（如果启用）
├── comparison_results.csv     # 对比结果（长格式）
├── comparison_wide.csv       # 对比结果（宽格式）
├── viz_trained/             # 训练模型的可视化结果
│   └── visualizations/
│       ├── CFFA/
│       ├── CFOCT/
│       └── OCTFA/
└── viz_baseline/            # Baseline模型的可视化结果（如果启用）
```

### 测试指标说明

| 指标 | 说明 |
|------|------|
| `num_samples` | 测试样本数量 |
| `auc@5/10/20` | AUC指标 |
| `mAUC` | 平均AUC |
| `combined_auc` | 综合AUC |
| `mse` | 均方误差（Acceptable样本） |
| `mace` | 角点平均误差（Acceptable样本） |

### 可视化文件说明

每个测试样本生成以下文件：

| 文件 | 说明 |
|------|------|
| `fix.png` | 固定图像 |
| `moving_original.png` | 移动图像（原始/形变后） |
| `moving_result.png` | 配准结果（使用预测的单应矩阵变换） |
| `moving_gt.png` | 移动图像GT（未形变） |
| `fix_with_kpts.png` | 固定图像+关键点 |
| `moving_with_kpts.png` | 移动图像+关键点 |
| `matches.png` | 匹配连线可视化 |
| `chessboard.png` | 棋盘格对比图 |
| `metrics.txt` | 指标记录 |

---

## 5. 数据集说明

### 训练集

- **260305_1_v30**: 生成的视网膜多模态配准数据
- 包含 CF-FA, CF-OCT, OCT-FA 模态对
- 应用域随机化增强

### 验证/测试集

- **CFFA**: CF（彩色眼底图像）与FA（荧光素血管造影）配对
- **CFOCT**: CF与OCT（光学相干断层扫描）配对  
- **OCTFA**: OCT与FA配对

---

## 6. 快速开始示例

### 完整训练+测试流程

```bash
# 1. 开始训练
cd /data/student/Fengjunming/MambaGlue-Retina2603
python scripts/v3/train_onMultiGen_vessels_enhanced.py \
    --name my_experiment \
    --batch_size 4 \
    --max_epochs 200 \
    --gpus 0

# 2. 测试训练好的模型
python scripts/v3/test_all_operationpre.py \
    --name my_experiment \
    --test_name final_test

# 3. 与Baseline对比
python scripts/v3/test_all_operationpre.py \
    --name my_experiment \
    --test_name compare_with_baseline \
    --baseline
```

---

## 7. 注意事项

1. **数据路径**: 代码会自动查找 `data/260305_1_v30` 和 `data/operation_pre_filtered_*` 目录
2. **GPU内存**: 如果遇到OOM，适当减小 `--batch_size` 或 `--img_size`
3. **预训练权重**: SuperPoint权重会自动从GitHub下载，如遇网络问题可手动下载放置
4. **域随机化**: `gen_data_enhance.py` 中的增强策略已经过温和调整，确保血管特征可见

---

## 8. 依赖项

- Python 3.8+
- PyTorch
- PyTorch Lightning
- OpenCV
- NumPy
- Matplotlib
- loguru (日志)
- mambaglue (已集成在项目中)
