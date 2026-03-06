import sys
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pprint
from pathlib import Path
from loguru import logger
import cv2
import numpy as np
import torch
import csv
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.strategies import DDPStrategy
import logging
from types import SimpleNamespace

# 添加父目录到 sys.path 以支持导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 导入 MambaGlue 相关模块
from mambaglue import MambaGlue, SuperPoint
from mambaglue import viz2d

# 导入真实数据集
from data.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset
from data.operation_pre_filtered_cfoct.operation_pre_filtered_cfoct_dataset import CFOCTDataset
from data.operation_pre_filtered_octfa.operation_pre_filtered_octfa_dataset import OCTFADataset

# 导入指标计算模块（v2_multi版本，对齐 metrics_cau_principle_0305.md）
from scripts.v2_multi.metrics import (
    compute_homography_errors,
    aggregate_metrics,
    set_metrics_verbose,
    error_auc,
    compute_auc_rop
)


def is_valid_homography(H, scale_min=0.1, scale_max=10.0, perspective_threshold=0.005):
    """单应矩阵防爆锁"""
    if H is None:
        return False
    if np.isnan(H).any() or np.isinf(H).any():
        return False

    det = np.linalg.det(H[:2, :2])
    if det < scale_min or det > scale_max:
        return False

    if abs(H[2, 0]) > perspective_threshold or abs(H[2, 1]) > perspective_threshold:
        return False

    return True


# ==========================================
# 配置函数
# ==========================================
def get_default_config():
    """获取默认配置"""
    conf = SimpleNamespace()
    conf.TRAINER = SimpleNamespace()
    conf.TRAINER.CANONICAL_BS = 4
    conf.TRAINER.CANONICAL_LR = 1e-4
    conf.TRAINER.TRUE_LR = 1e-4
    conf.TRAINER.RANSAC_PIXEL_THR = 3.0
    conf.TRAINER.SEED = 66
    conf.TRAINER.WORLD_SIZE = 1
    conf.TRAINER.TRUE_BATCH_SIZE = 4
    conf.TRAINER.PLOT_MODE = 'evaluation'

    conf.MATCHING = {
        'features': 'superpoint',
        'input_dim': 256,
        'descriptor_dim': 256,
        'depth_confidence': -1,
        'width_confidence': -1,
        'filter_threshold': 0.1,
        'flash': False
    }
    return conf

# ==========================================
# 工具函数
# ==========================================
def filter_valid_area(img1, img2):
    assert img1.shape[:2] == img2.shape[:2], "两张图片的尺寸必须一致"
    if len(img1.shape) == 3:
        mask1 = np.any(img1 > 10, axis=2)
    else:
        mask1 = img1 > 0
    if len(img2.shape) == 3:
        mask2 = np.any(img2 > 10, axis=2)
    else:
        mask2 = img2 > 0
    valid_mask = mask1 & mask2
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img1, img2
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    filtered_img1 = img1[row_min:row_max+1, col_min:col_max+1].copy()
    filtered_img2 = img2[row_min:row_max+1, col_min:col_max+1].copy()
    valid_mask_cropped = valid_mask[row_min:row_max+1, col_min:col_max+1]
    filtered_img1[~valid_mask_cropped] = 0
    filtered_img2[~valid_mask_cropped] = 0
    return filtered_img1, filtered_img2

def compute_corner_error(H_est, H_gt, height, width):
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    corners_homo = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    corners_gt_homo = (H_gt @ corners_homo.T).T
    corners_gt = corners_gt_homo[:, :2] / (corners_gt_homo[:, 2:] + 1e-6)
    corners_est_homo = (H_est @ corners_homo.T).T
    corners_est = corners_est_homo[:, :2] / (corners_est_homo[:, 2:] + 1e-6)
    try:
        errors = np.sqrt(np.sum((corners_est - corners_gt)**2, axis=1))
        mace = np.mean(errors)
    except:
        mace = float('inf')
    return mace

def create_chessboard(img1, img2, grid_size=4):
    H, W = img1.shape
    cell_h = H // grid_size
    cell_w = W // grid_size
    chessboard = np.zeros((H, W), dtype=img1.dtype)
    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w
            if (i + j) % 2 == 0:
                chessboard[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]
            else:
                chessboard[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
    return chessboard

# ==========================================
# 辅助类: RealDatasetWrapper
# ==========================================
class RealDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, split_name='test', dataset_name='MultiModal'):
        self.base_dataset = base_dataset
        self.split_name = split_name
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        moving_original_tensor = (moving_original_tensor + 1) / 2
        moving_gt_tensor = (moving_gt_tensor + 1) / 2

        if fix_tensor.shape[0] == 3:
            fix_gray = 0.299 * fix_tensor[0] + 0.587 * fix_tensor[1] + 0.114 * fix_tensor[2]
            fix_gray = fix_gray.unsqueeze(0)
        else:
            fix_gray = fix_tensor

        if moving_gt_tensor.shape[0] == 3:
            moving_gray = 0.299 * moving_gt_tensor[0] + 0.587 * moving_gt_tensor[1] + 0.114 * moving_gt_tensor[2]
            moving_gray = moving_gray.unsqueeze(0)
        else:
            moving_gray = moving_gt_tensor

        if moving_original_tensor.shape[0] == 3:
            moving_orig_gray = 0.299 * moving_original_tensor[0] + 0.587 * moving_original_tensor[1] + 0.114 * moving_original_tensor[2]
            moving_orig_gray = moving_orig_gray.unsqueeze(0)
        else:
            moving_orig_gray = moving_original_tensor

        fix_name = os.path.basename(fix_path)
        moving_name = os.path.basename(moving_path)

        try:
            T_fix_to_moving = torch.inverse(T_0to1)
        except:
            T_fix_to_moving = T_0to1

        return {
            'image0': fix_gray,
            'image1': moving_orig_gray,
            'image1_gt': moving_gray,
            'T_0to1': T_fix_to_moving,
            'pair_names': (fix_name, moving_name),
            'dataset_name': self.dataset_name,
            'split': self.split_name
        }

class TestDataModule:
    """测试用的数据模块，支持加载指定的数据集"""
    def __init__(self, args):
        self.args = args
        self.loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': True
        }

    def get_test_dataloader(self, datasets=None):
        """
        获取测试数据加载器
        datasets: list of dataset names to include, e.g., ['CFFA', 'CFOCT', 'OCTFA']
                 if None, load all datasets
        """
        script_dir = Path(__file__).parent.parent.parent
        val_dataset_list = []

        if datasets is None or 'CFFA' in datasets:
            cffa_dir = script_dir / 'data' / 'operation_pre_filtered_cffa'
            cffa_base = CFFADataset(root_dir=str(cffa_dir), split='val', mode='cf2fa')
            cffa_dataset = RealDatasetWrapper(cffa_base, split_name='test', dataset_name='CFFA')
            logger.info(f"加载 CFFA 测试集: {len(cffa_dataset)} 样本")
            val_dataset_list.append(cffa_dataset)

        if datasets is None or 'CFOCT' in datasets:
            cfoct_dir = script_dir / 'data' / 'operation_pre_filtered_cfoct'
            cfoct_base = CFOCTDataset(root_dir=str(cfoct_dir), split='val', mode='cf2oct')
            cfoct_dataset = RealDatasetWrapper(cfoct_base, split_name='test', dataset_name='CFOCT')
            logger.info(f"加载 CFOCT 测试集: {len(cfoct_dataset)} 样本")
            val_dataset_list.append(cfoct_dataset)

        if datasets is None or 'OCTFA' in datasets:
            octfa_dir = script_dir / 'data' / 'operation_pre_filtered_octfa'
            octfa_base = OCTFADataset(root_dir=str(octfa_dir), split='val', mode='fa2oct')
            octfa_dataset = RealDatasetWrapper(octfa_base, split_name='test', dataset_name='OCTFA')
            logger.info(f"加载 OCTFA 测试集: {len(octfa_dataset)} 样本")
            val_dataset_list.append(octfa_dataset)

        from torch.utils.data import ConcatDataset
        val_dataset = ConcatDataset(val_dataset_list)
        logger.info(f"测试集总样本数: {len(val_dataset)}")
        from torch.utils.data import DataLoader
        return DataLoader(val_dataset, shuffle=False, **self.loader_params)


def compute_metrics_for_dataset(evaluator, dataset_name):
    """为单个数据集计算指标"""
    if dataset_name in evaluator.per_dataset_errors and len(evaluator.per_dataset_errors[dataset_name]) > 0:
        errors = evaluator.per_dataset_errors[dataset_name]
        auc_dict = error_auc(errors, [5, 10, 20])
        mauc_dict = compute_auc_rop(errors, limit=25)

        mses = evaluator.per_dataset_mses.get(dataset_name, [])
        maces = evaluator.per_dataset_maces.get(dataset_name, [])

        return {
            'dataset': dataset_name,
            'auc@5': auc_dict.get('auc@5', 0.0),
            'auc@10': auc_dict.get('auc@10', 0.0),
            'auc@20': auc_dict.get('auc@20', 0.0),
            'mAUC': mauc_dict.get('mAUC', 0.0),
            'combined_auc': (auc_dict.get('auc@5', 0.0) + auc_dict.get('auc@10', 0.0) + auc_dict.get('auc@20', 0.0)) / 3.0,
            'mse': sum(mses) / len(mses) if mses else 0.0,
            'mace': sum(maces) / len(maces) if maces else 0.0,
            'num_samples': len(errors)
        }
    return None


class UnifiedEvaluator:
    """统一的评估器，支持按数据集分别计算指标"""
    def __init__(self, config=None):
        self.config = config
        self.reset()

    def reset(self):
        """重置累积的指标"""
        self.all_errors = []
        self.all_mses = []
        self.all_maces = []
        self.total_samples = 0
        self.failed_samples = 0
        self.inaccurate_samples = 0
        self.acceptable_samples = 0

        # 按数据集分别统计
        self.per_dataset_errors = {}
        self.per_dataset_mses = {}
        self.per_dataset_maces = {}
        self.per_dataset_samples = {}

    def evaluate_batch(self, batch, outputs, pl_module):
        """评估一个 batch"""
        matches0 = outputs['matches0']
        kpts0 = batch['keypoints0']
        kpts1 = batch['keypoints1']
        dataset_names = batch.get('dataset_name', ['unknown'] * kpts0.shape[0])

        B = kpts0.shape[0]

        mkpts0_f_list = []
        mkpts1_f_list = []
        m_bids_list = []

        for b in range(B):
            m0 = matches0[b]
            valid = m0 > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[valid]

            pts0 = kpts0[b][m_indices_0].cpu().numpy()
            pts1 = kpts1[b][m_indices_1].cpu().numpy()

            if len(pts0) > 0:
                mkpts0_f_list.append(torch.from_numpy(pts0).float())
                mkpts1_f_list.append(torch.from_numpy(pts1).float())
                m_bids_list.append(torch.full((len(pts0),), b, dtype=torch.long))

        metrics_batch = {
            'mkpts0_f': torch.cat(mkpts0_f_list, dim=0) if mkpts0_f_list else torch.empty(0, 2),
            'mkpts1_f': torch.cat(mkpts1_f_list, dim=0) if mkpts1_f_list else torch.empty(0, 2),
            'm_bids': torch.cat(m_bids_list, dim=0) if m_bids_list else torch.empty(0, dtype=torch.long),
            'T_0to1': batch['T_0to1'],
            'image0': batch['image0'],
            'dataset_name': batch['dataset_name']
        }

        compute_homography_errors(metrics_batch, self.config if self.config else pl_module.config)

        self.total_samples += B

        # 【修复】使用 metrics_batch 中的 auc_error 和 mse/mace 来判断样本状态
        # metrics.py 中：
        # - Failed: auc_error = 1e6
        # - Success (包括 inaccurate): auc_error = mace (有限值)
        # - Inaccurate: mse = inf
        # - Acceptable: mse = 有限值
        auc_error_list = metrics_batch.get('auc_error', [])
        mse_list = metrics_batch.get('mse', [])

        failed_count = 0
        inaccurate_count = 0
        acceptable_count = 0

        for i in range(B):
            if i < len(auc_error_list):
                auc_error = auc_error_list[i]
                # Failed 判断: auc_error = 1e6 (在 metrics.py 中设置的)
                if np.isclose(auc_error, 1e6):
                    failed_count += 1
                else:
                    # Success 样本，判断是 Inaccurate 还是 Acceptable
                    if i < len(mse_list) and np.isinf(mse_list[i]):
                        # mse = inf 表示 Inaccurate (mae > 50 或 mee > 20)
                        inaccurate_count += 1
                    else:
                        acceptable_count += 1
            else:
                # 如果没有 metrics 结果，保守起见计为 failed
                failed_count += 1

        self.failed_samples += failed_count
        self.inaccurate_samples += inaccurate_count
        self.acceptable_samples += acceptable_count

        if len(metrics_batch.get('auc_error', [])) > 0:
            self.all_errors.extend(list(metrics_batch['auc_error']))

        batch_mses = list(metrics_batch.get('mse', []))
        batch_maces = list(metrics_batch.get('mace', []))
        for mse in batch_mses:
            if np.isfinite(mse):
                self.all_mses.append(float(mse))
        for mace in batch_maces:
            if np.isfinite(mace):
                self.all_maces.append(float(mace))

        # 按数据集分别统计
        for b in range(B):
            dataset = dataset_names[b] if isinstance(dataset_names, list) else dataset_names
            if dataset not in self.per_dataset_errors:
                self.per_dataset_errors[dataset] = []
                self.per_dataset_mses[dataset] = []
                self.per_dataset_maces[dataset] = []
                self.per_dataset_samples[dataset] = 0

            self.per_dataset_samples[dataset] += 1

            # 样本级别的误差 - 使用 auc_error（用于 AUC 计算）
            if b < len(auc_error_list):
                self.per_dataset_errors[dataset].append(auc_error_list[b])

            # MSE 和 MACE - 仅统计 Acceptable 样本（过滤掉 inf）
            if b < len(batch_mses) and np.isfinite(batch_mses[b]):
                self.per_dataset_mses[dataset].append(float(batch_mses[b]))
            if b < len(batch_maces) and np.isfinite(batch_maces[b]):
                self.per_dataset_maces[dataset].append(float(batch_maces[b]))

        return {
            'H_est': metrics_batch.get('H_est', [np.eye(3)] * B),
            'mses': batch_mses,
            'maces': batch_maces,
            'metrics_batch': metrics_batch,
            'matches0': matches0,
            'kpts0': kpts0,
            'kpts1': kpts1
        }

    def compute_epoch_metrics(self):
        """计算整个 epoch 的聚合指标"""
        metrics = {}

        if self.all_errors and len(self.all_errors) > 0:
            auc_dict = error_auc(self.all_errors, [5, 10, 20])
            metrics['auc@5'] = auc_dict.get('auc@5', 0.0)
            metrics['auc@10'] = auc_dict.get('auc@10', 0.0)
            metrics['auc@20'] = auc_dict.get('auc@20', 0.0)

            mauc_dict = compute_auc_rop(self.all_errors, limit=25)
            metrics['mAUC'] = mauc_dict.get('mAUC', 0.0)
        else:
            metrics['auc@5'] = 0.0
            metrics['auc@10'] = 0.0
            metrics['auc@20'] = 0.0
            metrics['mAUC'] = 0.0

        metrics['combined_auc'] = (metrics['auc@5'] + metrics['auc@10'] + metrics['auc@20']) / 3.0

        metrics['mse'] = sum(self.all_mses) / len(self.all_mses) if self.all_mses else 0.0
        metrics['mace'] = sum(self.all_maces) / len(self.all_maces) if self.all_maces else 0.0
        metrics['inverse_mace'] = 1.0 / (1.0 + metrics['mace']) if metrics['mace'] > 0 else 1.0

        metrics['match_failure_rate'] = self.failed_samples / self.total_samples if self.total_samples > 0 else 0.0
        metrics['total_samples'] = self.total_samples
        metrics['failed_samples'] = self.failed_samples
        metrics['success_samples'] = self.total_samples - self.failed_samples
        metrics['inaccurate_samples'] = self.inaccurate_samples
        metrics['acceptable_samples'] = self.acceptable_samples
        metrics['inaccurate_rate'] = self.inaccurate_samples / self.total_samples if self.total_samples > 0 else 0.0
        metrics['acceptable_rate'] = self.acceptable_samples / self.total_samples if self.total_samples > 0 else 0.0

        # 按数据集分别计算指标
        metrics['per_dataset'] = {}
        for dataset_name in self.per_dataset_errors.keys():
            ds_metrics = compute_metrics_for_dataset(self, dataset_name)
            if ds_metrics:
                metrics['per_dataset'][dataset_name] = ds_metrics

        return metrics


def run_evaluation(pl_module, dataloader, config, verbose=True, save_visualizations=False, output_dir=None):
    """运行完整的评估流程"""
    evaluator = UnifiedEvaluator(config=config)

    pl_module.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            outputs = pl_module(batch)
            result = evaluator.evaluate_batch(batch, outputs, pl_module)

            if save_visualizations and output_dir:
                _visualize_batch(batch, result, output_dir, batch_idx)

            if verbose and batch_idx % 10 == 0:
                logger.info(f"已处理 {batch_idx + 1} 个 batch")

    metrics = evaluator.compute_epoch_metrics()

    if verbose:
        logger.info(f"评估完成: {metrics}")

    return metrics


def _visualize_batch(batch, outputs, output_dir, batch_idx):
    """可视化一个batch的结果"""
    import matplotlib.pyplot as plt
    from mambaglue import viz2d

    output_dir = Path(output_dir)
    batch_size = batch['image0'].shape[0]
    H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
    dataset_names = batch.get('dataset_name', ['unknown'] * batch_size)

    for sample_idx in range(batch_size):
        H_est = H_ests[sample_idx]

        if not is_valid_homography(H_est):
            H_est = np.eye(3)

        img0 = (batch['image0'][sample_idx, 0].cpu().numpy() * 255).astype(np.uint8)
        img1 = (batch['image1'][sample_idx, 0].cpu().numpy() * 255).astype(np.uint8)
        img1_gt = (batch['image1_gt'][sample_idx, 0].cpu().numpy() * 255).astype(np.uint8)

        h, w = img0.shape
        try:
            H_inv = np.linalg.inv(H_est)
            img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
        except:
            img1_result = img1.copy()

        dataset_name = dataset_names[sample_idx] if isinstance(dataset_names, list) else dataset_names

        pair_names = batch.get('pair_names', None)
        if pair_names:
            sample_name = f"{dataset_name}_batch{batch_idx:04d}_sample{sample_idx:02d}_{Path(pair_names[0][sample_idx]).stem}_vs_{Path(pair_names[1][sample_idx]).stem}"
        else:
            sample_name = f"{dataset_name}_batch{batch_idx:04d}_sample{sample_idx:02d}"

        save_path = output_dir / sample_name
        save_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_path / "fix.png"), img0)
        cv2.imwrite(str(save_path / "moving_original.png"), img1)
        cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
        cv2.imwrite(str(save_path / "moving_gt.png"), img1_gt)

        img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

        if 'kpts0' in outputs and 'kpts1' in outputs:
            kpts0_np = outputs['kpts0'][sample_idx].cpu().numpy()
            kpts1_np = outputs['kpts1'][sample_idx].cpu().numpy()

            for pt in kpts0_np:
                cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
            for pt in kpts1_np:
                cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)

            if 'matches0' in outputs:
                m0 = outputs['matches0'][sample_idx].cpu()
                valid = m0 > -1
                m_indices_0 = torch.where(valid)[0].numpy()
                m_indices_1 = m0[valid].numpy()

                for idx0 in m_indices_0:
                    pt = kpts0_np[idx0]
                    cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                for idx1 in m_indices_1:
                    pt = kpts1_np[idx1]
                    cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

                try:
                    fig = plt.figure(figsize=(12, 6))
                    viz2d.plot_images([img0, img1])
                    if len(m_indices_0) > 0:
                        viz2d.plot_matches(kpts0_np[m_indices_0], kpts1_np[m_indices_1], color='lime', lw=0.5)
                    plt.savefig(str(save_path / "matches.png"), bbox_inches='tight', dpi=100)
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"绘制匹配图失败: {e}")

        cv2.imwrite(str(save_path / "fix_with_kpts.png"), img0_color)
        cv2.imwrite(str(save_path / "moving_with_kpts.png"), img1_color)

        try:
            cb = create_chessboard(img1_result, img0)
            cv2.imwrite(str(save_path / "chessboard.png"), cb)
        except:
            pass

        if 'mses' in outputs and sample_idx < len(outputs['mses']):
            mse = outputs['mses'][sample_idx]
            mace = outputs['maces'][sample_idx]
            with open(save_path / "metrics.txt", "w") as f:
                f.write(f"MSE: {mse:.6f}\n")
                f.write(f"MACE: {mace:.4f}\n")
                if 'matches0' in outputs:
                    m0 = outputs['matches0'][sample_idx].cpu()
                    valid = m0 > -1
                    num_matches = torch.sum(valid).item()
                    f.write(f"Matches: {num_matches}\n")




# ==========================================
# 核心模型: PL_MambaGlue
# ==========================================
class PL_MambaGlue(pl.LightningModule):
    def __init__(self, config, result_dir=None):
        super().__init__()
        self.config = config
        self.result_dir = result_dir
        self.save_hyperparameters({'config': str(config)})

        self.extractor = SuperPoint(max_num_keypoints=2048).eval()
        sp_url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"
        try:
            sp_state = torch.hub.load_state_dict_from_url(sp_url, map_location='cpu')
            self.extractor.load_state_dict(sp_state, strict=False)
            logger.info("成功加载 SuperPoint 预训练权重")
        except Exception as e:
            logger.warning(f"加载 SuperPoint 预训练权重失败: {e}，使用随机初始化")

        for param in self.extractor.parameters():
            param.requires_grad = False

        mg_conf = config.MATCHING.copy()
        self.matcher = MambaGlue(**mg_conf)

        self.force_viz = False
        self._test_step_errors = []
        self._test_step_mse = []
        self._test_step_mace = []

    def configure_optimizers(self):
        lr = self.config.TRAINER.TRUE_LR
        optimizer = torch.optim.Adam(self.matcher.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "combined_auc",
            },
        }

    def forward(self, batch):
        with torch.no_grad():
            if 'keypoints0' not in batch:
                feats0 = self.extractor({'image': batch['image0']})
                feats1 = self.extractor({'image': batch['image1']})
                batch.update({
                    'keypoints0': feats0['keypoints'],
                    'descriptors0': feats0['descriptors'],
                    'scores0': feats0['keypoint_scores'],
                    'keypoints1': feats1['keypoints'],
                    'descriptors1': feats1['descriptors'],
                    'scores1': feats1['keypoint_scores']
                })

        data = {
            'image0': {
                'keypoints': batch['keypoints0'],
                'descriptors': batch['descriptors0'],
                'image': batch['image0']
            },
            'image1': {
                'keypoints': batch['keypoints1'],
                'descriptors': batch['descriptors1'],
                'image': batch['image1']
            }
        }
        return self.matcher(data)

    def _compute_gt_matches(self, kpts0, kpts1, T_0to1, dist_th=3.0):
        B, M, _ = kpts0.shape
        B, N, _ = kpts1.shape
        device = kpts0.device
        kpts0_h = torch.cat([kpts0, torch.ones(B, M, 1, device=device)], dim=-1)
        kpts0_warped_h = torch.matmul(kpts0_h, T_0to1.transpose(1, 2))
        kpts0_warped = kpts0_warped_h[..., :2] / (kpts0_warped_h[..., 2:] + 1e-8)
        dist = torch.cdist(kpts0_warped, kpts1)
        min_dist, matched_indices = torch.min(dist, dim=-1)
        mask = min_dist < dist_th
        matches_gt = torch.where(mask, matched_indices, torch.tensor(-1, device=device))
        return matches_gt

    def _compute_loss(self, outputs, kpts0, kpts1, T_0to1):
        scores = outputs['log_assignment']
        matches_gt = self._compute_gt_matches(kpts0, kpts1, T_0to1)
        B, M, N = scores.shape[0], scores.shape[1]-1, scores.shape[2]-1
        targets = matches_gt.clone()
        targets[targets == -1] = N
        target_log_probs = torch.gather(scores[:, :M, :], 2, targets.unsqueeze(2)).squeeze(2)
        loss = -target_log_probs.mean()
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self._compute_loss(outputs, batch['keypoints0'], batch['keypoints1'], batch['T_0to1'])
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        matches0 = outputs['matches0']
        kpts0 = batch['keypoints0']
        kpts1 = batch['keypoints1']

        B = kpts0.shape[0]
        H_ests = []
        mkpts0_f_list = []
        mkpts1_f_list = []
        m_bids_list = []

        for b in range(B):
            m0 = matches0[b]
            valid = m0 > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[valid]

            pts0 = kpts0[b][m_indices_0].cpu().numpy()
            pts1 = kpts1[b][m_indices_1].cpu().numpy()

            if len(pts0) > 0:
                mkpts0_f_list.append(torch.from_numpy(pts0).float())
                mkpts1_f_list.append(torch.from_numpy(pts1).float())
                m_bids_list.append(torch.full((len(pts0),), b, dtype=torch.long))

            if len(pts0) >= 4:
                try:
                    H, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, self.config.TRAINER.RANSAC_PIXEL_THR)
                    if H is None:
                        H = np.eye(3)
                except:
                    H = np.eye(3)
            else:
                H = np.eye(3)
            H_ests.append(H)

        metrics_batch = {
            'mkpts0_f': torch.cat(mkpts0_f_list, dim=0) if mkpts0_f_list else torch.empty(0, 2),
            'mkpts1_f': torch.cat(mkpts1_f_list, dim=0) if mkpts1_f_list else torch.empty(0, 2),
            'm_bids': torch.cat(m_bids_list, dim=0) if m_bids_list else torch.empty(0, dtype=torch.long),
            'T_0to1': batch['T_0to1'],
            'image0': batch['image0'],
            'dataset_name': batch['dataset_name']
        }

        set_metrics_verbose(True)
        compute_homography_errors(metrics_batch, self.config)

        # 直接使用 metrics.py 返回的 H_est（经过 Spatial Binning）
        # 注意：metrics.py 中当 num_matches < 4 时会跳过，所以长度可能不够
        H_ests_raw = metrics_batch.get('H_est', [])
        # 确保 H_ests 长度与 batch size 一致
        H_ests = list(H_ests_raw)
        while len(H_ests) < B:
            H_ests.append(np.eye(3))  # 填充单位矩阵

        if len(metrics_batch.get('auc_error', [])) > 0:
            self._test_step_errors.extend(metrics_batch['auc_error'])

        if len(metrics_batch.get('mse', [])) > 0:
            for m in metrics_batch['mse']:
                if not np.isinf(m):
                    self._test_step_mse.append(m)
        if len(metrics_batch.get('mace', [])) > 0:
            for m in metrics_batch['mace']:
                if not np.isinf(m):
                    self._test_step_mace.append(m)

        return {
            'H_est': H_ests,
            'kpts0': kpts0,
            'kpts1': kpts1,
            'matches0': matches0,
            'metrics_batch': metrics_batch
        }

    def on_test_epoch_start(self):
        self._test_step_errors = []
        self._test_step_mse = []
        self._test_step_mace = []

    def on_test_epoch_end(self):
        if self._test_step_errors and len(self._test_step_errors) > 0:
            auc_dict = error_auc(self._test_step_errors, [5, 10, 20])
            auc5 = auc_dict.get('auc@5', 0.0)
            auc10 = auc_dict.get('auc@10', 0.0)
            auc20 = auc_dict.get('auc@20', 0.0)
            mauc_dict = compute_auc_rop(self._test_step_errors, limit=25)
            mauc = mauc_dict.get('mAUC', 0.0)
        else:
            auc5 = auc10 = auc20 = mauc = 0.0

        combined_auc = (auc5 + auc10 + auc20) / 3.0
        
        # 【修复】当没有 Acceptable 样本时，应该返回 NaN 而不是 0
        # 因为 0 会误导用户以为计算成功了
        avg_mse = np.mean(self._test_step_mse) if self._test_step_mse else float('nan')
        avg_mace = np.mean(self._test_step_mace) if self._test_step_mace else float('nan')

        self.log('auc@5',        auc5,         on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('auc@10',       auc10,        on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('auc@20',       auc20,        on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('mAUC',         mauc,         on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('combined_auc', combined_auc, on_epoch=True, prog_bar=True,  logger=True, sync_dist=False)
        self.log('mse', avg_mse, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('mace', avg_mace, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)

# ==========================================
# 回调逻辑: TestCallback
# ==========================================
class TestCallback(Callback):
    def __init__(self, args, output_dir):
        super().__init__()
        self.args = args
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.total_samples = 0
        self.failed_samples = 0
        self.inaccurate_samples = 0
        self.acceptable_samples = 0

        import csv
        self.csv_path = self.output_dir / "test_metrics.csv"
        with open(self.csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Batch", "Test Loss", "MSE", "MACE", "AUC@5", "AUC@10", "AUC@20", "mAUC", "Failed", "Inaccurate", "Acceptable"])

    def on_test_epoch_start(self, trainer, pl_module):
        self.total_samples = 0
        self.failed_samples = 0
        self.inaccurate_samples = 0
        self.acceptable_samples = 0

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._process_batch(trainer, pl_module, batch, outputs, batch_idx, save_images=True)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.total_samples == 0:
            logger.warning("没有收集到任何测试指标")
            return

        metrics = trainer.callback_metrics

        if hasattr(pl_module, '_test_step_errors') and pl_module._test_step_errors:
            errors = pl_module._test_step_errors
            auc_dict = error_auc(errors, [5, 10, 20])
            auc5 = auc_dict.get('auc@5', 0.0)
            auc10 = auc_dict.get('auc@10', 0.0)
            auc20 = auc_dict.get('auc@20', 0.0)
            mauc_dict = compute_auc_rop(errors, limit=25)
            mauc = mauc_dict.get('mAUC', 0.0)
        else:
            auc5 = auc10 = auc20 = mauc = 0.0

        combined_auc = (auc5 + auc10 + auc20) / 3.0

        # 从 metrics 中获取 MSE 和 MACE
        # 注意：如果没有 Acceptable 样本，PL_MambaGlue 会返回 NaN
        avg_mse = metrics.get('mse', 0.0)
        if hasattr(avg_mse, 'item'):
            avg_mse = avg_mse.item()
        # 如果是 0 但没有 Acceptable 样本，可能是没有数据，此时显示为 NaN 更合理
        if avg_mse == 0.0 and not hasattr(pl_module, '_test_step_mse'):
            avg_mse = float('nan')
            
        avg_mace = metrics.get('mace', 0.0)
        if hasattr(avg_mace, 'item'):
            avg_mace = avg_mace.item()
        if avg_mace == 0.0 and not hasattr(pl_module, '_test_step_mace'):
            avg_mace = float('nan')

        match_failure_rate = self.failed_samples / self.total_samples if self.total_samples > 0 else 0.0

        metric_str = f"mse: {avg_mse:.4f} | mace: {avg_mace:.4f} | auc@5: {auc5:.4f} | auc@10: {auc10:.4f} | auc@20: {auc20:.4f} | mAUC: {mauc:.4f} | combined_auc: {combined_auc:.4f}"
        logger.info(f"测试总结 >> {metric_str}")
        logger.info(f"总样本数: {self.total_samples}, 失败: {self.failed_samples}, 不准确: {self.inaccurate_samples}, 可接受: {self.acceptable_samples}")

        summary_path = self.output_dir / "test_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"测试总结\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"测试损失: {metrics.get('test_loss', 0.0):.6f}\n")
            f.write(f"总样本数: {self.total_samples}\n")
            f.write(f"失败样本数: {self.failed_samples}\n")
            f.write(f"不准确样本数: {self.inaccurate_samples}\n")
            f.write(f"可接受样本数: {self.acceptable_samples}\n")
            f.write(f"匹配失败率: {match_failure_rate:.4f}\n")
            f.write(f"MSE (仅Acceptable): {avg_mse:.6f}\n")
            f.write(f"MACE (仅Acceptable): {avg_mace:.4f}\n")
            f.write(f"AUC@5: {auc5:.4f}\n")
            f.write(f"AUC@10: {auc10:.4f}\n")
            f.write(f"AUC@20: {auc20:.4f}\n")
            f.write(f"mAUC: {mauc:.4f}\n")
            f.write(f"Combined AUC: {combined_auc:.4f}\n")

        logger.info(f"测试总结已保存到: {summary_path}")

    def _process_batch(self, trainer, pl_module, batch, outputs, batch_idx, save_images=False):
        """
        处理测试批次，直接使用 metrics.py 的计算结果
        严格按照 metrics_cau_principle_0304.md 的逻辑进行统计
        """
        batch_size = batch['image0'].shape[0]
        H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
        metrics_batch = outputs.get('metrics_batch', {})

        # 直接使用 metrics.py 计算的 avg_dist 来判断 failed/inaccurate
        # metrics.py 中：
        # - Failed: avg_dist = 1e6
        # - Success (包括 inaccurate): avg_dist = 实际计算的 avg_dist
        mse_list = metrics_batch.get('mse', [])
        mace_list = metrics_batch.get('mace', [])
        auc_error_list = metrics_batch.get('auc_error', [])

        for i in range(batch_size):
            self.total_samples += 1

            # 直接使用 metrics.py 的判断结果
            if i < len(auc_error_list):
                auc_error = auc_error_list[i]

                # Failed 判断: error = 1e6 (在 metrics.py 中设置的)
                if np.isclose(auc_error, 1e6):
                    self.failed_samples += 1
                else:
                    # Success 样本，判断是 Inaccurate 还是 Acceptable
                    if i < len(mse_list) and np.isinf(mse_list[i]):
                        # mse = inf 表示 Inaccurate (mae > 50 或 mee > 20)
                        self.inaccurate_samples += 1
                    else:
                        self.acceptable_samples += 1
            else:
                # 如果没有 metrics 结果，保守起见计为 failed
                self.failed_samples += 1

            # 后续代码保持不变，用于可视化
            H_est = H_ests[i]

            img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1_gt = (batch['image1_gt'][i, 0].cpu().numpy() * 255).astype(np.uint8)

            h, w = img0.shape
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
            except:
                img1_result = img1.copy()

            if save_images:
                sample_name = f"batch{batch_idx:04d}_sample{i:02d}_{Path(batch['pair_names'][0][i]).stem}_vs_{Path(batch['pair_names'][1][i]).stem}"
                save_path = self.output_dir / sample_name
                save_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path / "fix.png"), img0)
                cv2.imwrite(str(save_path / "moving_original.png"), img1)
                cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
                cv2.imwrite(str(save_path / "moving_gt.png"), img1_gt)

                img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
                img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

                if 'kpts0' in outputs and 'kpts1' in outputs:
                    kpts0_np = outputs['kpts0'][i].cpu().numpy()
                    kpts1_np = outputs['kpts1'][i].cpu().numpy()

                    for pt in kpts0_np:
                        cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
                    for pt in kpts1_np:
                        cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)

                    if 'matches0' in outputs:
                        m0 = outputs['matches0'][i].cpu()
                        valid = m0 > -1
                        m_indices_0 = torch.where(valid)[0].numpy()
                        m_indices_1 = m0[valid].numpy()

                        for idx0 in m_indices_0:
                            pt = kpts0_np[idx0]
                            cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                        for idx1 in m_indices_1:
                            pt = kpts1_np[idx1]
                            cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

                        try:
                            fig = plt.figure(figsize=(12, 6))
                            viz2d.plot_images([img0, img1])
                            if len(m_indices_0) > 0:
                                viz2d.plot_matches(kpts0_np[m_indices_0], kpts1_np[m_indices_1], color='lime', lw=0.5)
                            plt.savefig(str(save_path / "matches.png"), bbox_inches='tight', dpi=100)
                            plt.close(fig)
                        except Exception as e:
                            logger.warning(f"绘制匹配图失败: {e}")

                cv2.imwrite(str(save_path / "fix_with_kpts.png"), img0_color)
                cv2.imwrite(str(save_path / "moving_with_kpts.png"), img1_color)

                try:
                    cb = create_chessboard(img1_result, img0)
                    cv2.imwrite(str(save_path / "chessboard.png"), cb)
                except:
                    pass

# ==========================================
# 训练脚本配置函数
# ==========================================
def get_train_script_config(train_script):
    """根据训练脚本名称返回对应的配置"""
    configs = {
        'train_onPureGen_v2': {
            'import_path': 'scripts.v2_multi.train_onPureGen_v2',
            'class_name': 'PL_MambaGlue_Gen',
            'result_dir': 'mambaglue_gen_{train_mode}'
        },
        'train_onReal': {
            'import_path': 'scripts.v2_multi.train_onReal',
            'class_name': 'PL_MambaGlue_Real',
            'result_dir': 'mambaglue_{train_mode}',
            'use_train_mode': True
        },
        'train_onMultiGen_vessels_enhanced': {
            'import_path': 'scripts.v2_multi.train_onMultiGen_vessels_enhanced',
            'class_name': 'PL_MambaGlue_Gen',
            'result_dir': 'mambaglue_gen'
        }
    }
    return configs.get(train_script, None)

# ==========================================
# 参数解析和主函数
# ==========================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MambaGlue 统一测试脚本")

    # 支持的训练脚本
    parser.add_argument('--train_script', '-s', type=str, required=True,
                        choices=['train_onPureGen_v2', 'train_onReal', 'train_onMultiGen_vessels_enhanced'],
                        help='训练脚本名称')

    # train_onPureGen_v2 专用参数
    parser.add_argument('--train_mode', '-m', type=str, default='mixed',
                        choices=['cffa', 'cfoct', 'octfa', 'mixed'],
                        help='训练模式: cffa, cfoct, octfa (train_onReal仅支持这三个)')

    # 测试数据集选择 (用于混合模式测试时指定数据集)
    parser.add_argument('--test_datasets', '-d', type=str, default=None,
                        help='指定测试数据集，用逗号分隔，如 "CFFA,CFOCT,OCTFA" 或 "CFFA"')

    parser.add_argument('--name', '-n', type=str, required=True,
                        help='模型名称（用于定位结果目录）')
    parser.add_argument('--test_name', '-t', type=str, required=True,
                        help='测试名称（结果保存在结果目录下）')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='检查点路径（默认使用 best_checkpoint/model.ckpt）')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--gpus', type=str, default='0', help='GPU设备ID')
    parser.add_argument('--img_size', type=int, default=512, help='图像大小')
    parser.add_argument('--no_viz', action='store_true', help='禁用可视化')

    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()

    # 获取训练脚本配置
    script_config = get_train_script_config(args.train_script)
    if script_config is None:
        logger.error(f"未知的训练脚本: {args.train_script}")
        return

    # train_onReal 不支持 mixed 模式，默认设为 cffa
    if args.train_script == 'train_onReal' and args.train_mode == 'mixed':
        logger.warning("train_onReal 不支持 mixed 模式，默认使用 cffa")
        args.train_mode = 'cffa'

    # 动态导入模块
    import importlib
    module = importlib.import_module(script_config['import_path'])
    pl_class = getattr(module, script_config['class_name'])
    get_default_config = getattr(module, 'get_default_config')

    # 获取配置
    config = get_default_config()
    pl.seed_everything(config.TRAINER.SEED)

    # 确定结果目录和checkpoint路径
    if args.train_script == 'train_onPureGen_v2' or script_config.get('use_train_mode', False):
        mode_dir = script_config['result_dir'].format(train_mode=args.train_mode)
    else:
        mode_dir = script_config['result_dir']

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path(f"results/{mode_dir}/{args.name}/best_checkpoint/model.ckpt")

    if not ckpt_path.exists():
        logger.error(f"检查点不存在: {ckpt_path}")
        logger.info(f"请确保训练模型存在，或使用 --checkpoint 指定有效的检查点路径")
        return

    logger.info(f"加载检查点: {ckpt_path}")

    # 设置输出目录
    output_dir = Path(f"results/{mode_dir}/{args.name}/{args.test_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置日志
    log_file = output_dir / "test_log.txt"
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level="INFO")
    logger.add(log_file, format=log_format, level="INFO", mode="w")
    logger.info(f"日志将保存到: {log_file}")

    # GPU配置
    if ',' in str(args.gpus):
        gpus_list = [int(x) for x in args.gpus.split(',')]
        _n_gpus = len(gpus_list)
    else:
        try:
            gpus_list = [int(args.gpus)]
            _n_gpus = 1
        except:
            gpus_list = 'auto'
            _n_gpus = 1

    config.TRAINER.WORLD_SIZE = max(_n_gpus, 1)
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size

    logger.info(f"训练脚本: {args.train_script}")
    if args.train_script == 'train_onPureGen_v2':
        logger.info(f"训练模式: {args.train_mode}")
    logger.info(f"模型名称: {args.name}")
    logger.info(f"测试名称: {args.test_name}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"GPU配置: devices={gpus_list}, num_gpus={_n_gpus}")

    # 从检查点加载模型
    model = pl_class.load_from_checkpoint(
        str(ckpt_path),
        config=config,
        result_dir=str(output_dir)
    )
    model.eval()

    # 初始化测试数据模块
    test_dm = TestDataModule(args)

    # 确定测试数据集
    # 优先使用命令行显式指定的 -d / --test_datasets
    # 否则，对于按模态训练的脚本（train_onPureGen_v2, train_onReal），默认只在对应模态上测试
    test_datasets = None
    if args.test_datasets:
        test_datasets = [ds.strip() for ds in args.test_datasets.split(',')]
        logger.info(f"指定测试数据集: {test_datasets}")
    else:
        # 未显式指定时，根据 train_mode 选择默认测试集
        if args.train_script in ['train_onPureGen_v2', 'train_onReal']:
            mode2datasets = {
                'cffa': ['CFFA'],
                'cfoct': ['CFOCT'],
                'octfa': ['OCTFA'],
                'mixed': ['CFFA', 'CFOCT', 'OCTFA'],
            }
            test_datasets = mode2datasets.get(args.train_mode, ['CFFA', 'CFOCT', 'OCTFA'])
            logger.info(f"根据 train_mode 自动选择测试数据集: {test_datasets}")

    test_dataloader = test_dm.get_test_dataloader(datasets=test_datasets)

    logger.info(f"开始测试 (训练脚本: {args.train_script} | 模型: {args.name})")

    # 运行评估
    set_metrics_verbose(True)
    metrics = run_evaluation(
        model,
        test_dataloader,
        config=config,
        verbose=True,
        save_visualizations=not args.no_viz,
        output_dir=output_dir
    )

    # 保存测试总结
    summary_path = output_dir / "test_summary.txt"
    with open(summary_path, "w") as f:
        f.write("测试总结\n")
        f.write("=" * 50 + "\n")
        f.write(f"Train Script: {args.train_script}\n")
        if args.train_script == 'train_onPureGen_v2':
            f.write(f"Train Mode: {args.train_mode}\n")
        f.write(f"Test Name: {args.test_name}\n")
        f.write(f"Model Name: {args.name}\n")
        if args.test_datasets:
            f.write(f"Test Datasets: {args.test_datasets}\n")
        f.write(f"\n--- Overall Metrics ---\n")
        f.write(f"总样本数: {metrics['total_samples']}\n")
        f.write(f"匹配成功样本数: {metrics['success_samples']}\n")
        f.write(f"匹配失败样本数: {metrics['failed_samples']}\n")
        f.write(f"匹配失败率: {metrics['match_failure_rate']:.4f}\n")
        f.write(f"Inaccurate 样本数: {metrics['inaccurate_samples']}\n")
        f.write(f"Acceptable 样本数: {metrics['acceptable_samples']}\n")
        f.write(f"MSE (仅 Acceptable): {metrics['mse']:.6f}\n")
        f.write(f"MACE (仅 Acceptable): {metrics['mace']:.4f}\n")
        f.write(f"AUC@5: {metrics['auc@5']:.4f}\n")
        f.write(f"AUC@10: {metrics['auc@10']:.4f}\n")
        f.write(f"AUC@20: {metrics['auc@20']:.4f}\n")
        f.write(f"mAUC: {metrics['mAUC']:.4f}\n")
        f.write(f"Combined AUC: {metrics['combined_auc']:.4f}\n")
        f.write(f"Inverse MACE: {metrics['inverse_mace']:.6f}\n")

        # 按数据集分别输出
        if 'per_dataset' in metrics and metrics['per_dataset']:
            f.write(f"\n--- Per-Dataset Metrics ---\n")
            for ds_name, ds_metrics in metrics['per_dataset'].items():
                f.write(f"\n{ds_name}:\n")
                f.write(f"  样本数: {ds_metrics['num_samples']}\n")
                f.write(f"  AUC@5: {ds_metrics['auc@5']:.4f}\n")
                f.write(f"  AUC@10: {ds_metrics['auc@10']:.4f}\n")
                f.write(f"  AUC@20: {ds_metrics['auc@20']:.4f}\n")
                f.write(f"  mAUC: {ds_metrics['mAUC']:.4f}\n")
                f.write(f"  Combined AUC: {ds_metrics['combined_auc']:.4f}\n")
                f.write(f"  MSE: {ds_metrics['mse']:.6f}\n")
                f.write(f"  MACE: {ds_metrics['mace']:.4f}\n")

    # 保存 CSV 格式的汇总结果
    csv_path = output_dir / "test_results.csv"
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Dataset', 'Samples', 'AUC@5', 'AUC@10', 'AUC@20', 'mAUC', 'Combined_AUC', 'MSE', 'MACE'])

        # 写入总体结果
        writer.writerow([
            'Overall',
            metrics['total_samples'],
            f"{metrics['auc@5']:.4f}",
            f"{metrics['auc@10']:.4f}",
            f"{metrics['auc@20']:.4f}",
            f"{metrics['mAUC']:.4f}",
            f"{metrics['combined_auc']:.4f}",
            f"{metrics['mse']:.6f}",
            f"{metrics['mace']:.4f}"
        ])

        # 写入各数据集结果
        if 'per_dataset' in metrics and metrics['per_dataset']:
            for ds_name, ds_metrics in metrics['per_dataset'].items():
                writer.writerow([
                    ds_name,
                    ds_metrics['num_samples'],
                    f"{ds_metrics['auc@5']:.4f}",
                    f"{ds_metrics['auc@10']:.4f}",
                    f"{ds_metrics['auc@20']:.4f}",
                    f"{ds_metrics['mAUC']:.4f}",
                    f"{ds_metrics['combined_auc']:.4f}",
                    f"{ds_metrics['mse']:.6f}",
                    f"{ds_metrics['mace']:.4f}"
                ])

    logger.info(f"测试总结已保存到: {summary_path}")
    logger.info(f"CSV结果已保存到: {csv_path}")
    logger.info(f"测试完成! 结果已保存到: {output_dir}")

if __name__ == '__main__':
    main()
