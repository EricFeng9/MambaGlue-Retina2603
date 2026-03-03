"""
域随机化增强模块 (Domain Randomization Enhancement)

目的: 打破生成数据中不同模态之间的纹理相关性
     迫使模型学习真正的跨模态不变性特征 (几何结构) 而非表面纹理

核心策略 (参考 SynthMorph):
  采用适度的域随机化，破坏纹理捷径的同时保留血管特征可见性

增强策略分为四个强度等级（课程学习）：
  - 不增强 (40% 概率): 原始图像
  - 弱增强 (30% 概率): 只有颜色空间变换
  - 中等增强 (20% 概率): 颜色 + 轻微噪声
  - 强增强 (10% 概率): 全部增强

【关键】fix 和 moving 使用完全独立的随机参数
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random

# --- 不同强度的域随机化 ---

def apply_weak_augmentation(img_tensor):
    """
    弱增强：只有颜色空间变换
    - Gamma 变换 (0.7-1.3，温和范围)
    - 对比度调整 (0.8-1.2x，温和范围)
    """
    B, C, H, W = img_tensor.shape
    
    # Gamma 变换 (50% 概率)
    if random.random() < 0.5:
        gamma = random.uniform(0.7, 1.3)
        img_tensor = torch.pow(img_tensor.clamp(1e-6, 1.0), gamma)
    
    # 对比度调整
    contrast = random.uniform(0.8, 1.2)
    for b in range(B):
        mean_val = img_tensor[b].mean()
        img_tensor[b] = (img_tensor[b] - mean_val) * contrast + mean_val
    
    return img_tensor.clamp(0, 1)


def apply_medium_augmentation(img_tensor):
    """
    中等增强：颜色变换 + 轻微噪声
    - 图像反色 (10% 概率)
    - Gamma 变换 (0.6-1.5)
    - 对比度调整 (0.7-1.4x)
    - 亮度偏移 (±0.05，很小)
    - 全局噪声 (10% 概率，很小)
    """
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    
    # 图像反色 (10% 概率)
    if random.random() < 0.1:
        img_tensor = 1.0 - img_tensor
    
    # Gamma 变换 (50% 概率)
    if random.random() < 0.5:
        gamma = random.uniform(0.6, 1.5)
        img_tensor = torch.pow(img_tensor.clamp(1e-6, 1.0), gamma)
    
    # 对比度调整
    contrast = random.uniform(0.7, 1.4)
    for b in range(B):
        mean_val = img_tensor[b].mean()
        img_tensor[b] = (img_tensor[b] - mean_val) * contrast + mean_val
    img_tensor = img_tensor.clamp(0, 1)
    
    # 亮度偏移 (20% 概率)
    if random.random() < 0.2:
        brightness = random.uniform(-0.05, 0.05)
        img_tensor = (img_tensor + brightness).clamp(0, 1)
    
    # 全局噪声 (10% 概率)
    if random.random() < 0.1:
        noise_std = random.uniform(0.002, 0.005)
        noise = torch.randn_like(img_tensor) * noise_std
        img_tensor = (img_tensor + noise).clamp(0, 1)
    
    return img_tensor


def apply_strong_augmentation(img_tensor):
    """
    强增强：全部增强策略
    - 图像反色 (25% 概率)
    - Gamma 变换 (0.5-2.0)
    - 对比度调整 (0.6-1.5x)
    - 亮度偏移 (±0.1)
    - 黑色条纹 (5% 概率)
    - 不均匀噪声 (30% 概率)
    - 全局噪声 (10% 概率)
    """
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    
    # 图像反色 (25% 概率)
    if random.random() < 0.25:
        img_tensor = 1.0 - img_tensor
    
    # Gamma 变换 (50% 概率)
    if random.random() < 0.5:
        gamma = random.uniform(0.5, 2.0)
        img_tensor = torch.pow(img_tensor.clamp(1e-6, 1.0), gamma)
    
    # 对比度调整
    contrast = random.uniform(0.6, 1.5)
    for b in range(B):
        mean_val = img_tensor[b].mean()
        img_tensor[b] = (img_tensor[b] - mean_val) * contrast + mean_val
    img_tensor = img_tensor.clamp(0, 1)
    
    # 亮度偏移 (30% 概率)
    if random.random() < 0.3:
        brightness = random.uniform(-0.1, 0.1)
        img_tensor = (img_tensor + brightness).clamp(0, 1)
    
    # 黑色条纹 (5% 概率)
    if random.random() < 0.05:
        num_stripes = random.randint(1, 2)
        mask = torch.ones_like(img_tensor)
        for _ in range(num_stripes):
            is_vertical = random.random() < 0.5
            width = random.randint(1, 2)
            if is_vertical:
                pos = random.randint(0, max(1, W - width - 1))
                mask[:, :, :, pos:pos+width] = 0.0
            else:
                pos = random.randint(0, max(1, H - width - 1))
                mask[:, :, pos:pos+width, :] = 0.0
        img_tensor = img_tensor * mask
    
    # 不均匀噪声 (30% 概率)
    if random.random() < 0.3:
        noise_variance_map = torch.rand(1, 1, 4, 4, device=device)
        noise_variance_map = torch.where(
            noise_variance_map > 0.75,
            noise_variance_map,
            torch.zeros_like(noise_variance_map)
        )
        noise_variance_map = F.interpolate(
            noise_variance_map,
            size=(H, W),
            mode='bicubic',
            align_corners=False
        )
        noise_variance_map = noise_variance_map.expand(B, 1, H, W)
        
        noise = torch.randn_like(img_tensor) * (noise_variance_map * 0.015)
        darkness = noise_variance_map * 0.1
        img_tensor = img_tensor - darkness + noise
        img_tensor = img_tensor.clamp(0, 1)
    
    # 全局噪声 (10% 概率)
    if random.random() < 0.1:
        noise_std = random.uniform(0.002, 0.008)
        noise = torch.randn_like(img_tensor) * noise_std
        img_tensor = (img_tensor + noise).clamp(0, 1)
    
    # 动态范围保护
    for b in range(B):
        img_min = img_tensor[b].min()
        img_max = img_tensor[b].max()
        dynamic_range = img_max - img_min
        
        if dynamic_range < 0.15:
            img_tensor[b] = (img_tensor[b] - img_min) / (dynamic_range + 1e-6)
            img_tensor[b] = img_tensor[b] * 0.9 + 0.05
        elif dynamic_range < 0.3:
            img_tensor[b] = (img_tensor[b] - img_min) / (dynamic_range + 1e-6)
            img_tensor[b] = img_tensor[b] * 0.8 + 0.1
    
    return img_tensor.clamp(0, 1)


def random_domain_augment_image(image):
    """
    对单张图像应用域随机化（渐进式强度）
    
    按概率选择增强强度：
    - 40% 不增强
    - 30% 弱增强
    - 20% 中等增强
    - 10% 强增强
    
    Args:
        image: Tensor [C, H, W] 或 [H, W]，值域 [0, 1]
    
    Returns:
        增强后的图像 tensor
    """
    is_numpy = isinstance(image, np.ndarray)
    
    # 转换为 tensor
    if is_numpy:
        img_dtype = image.dtype
        if image.ndim == 2:
            img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        else:
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            
        if img_tensor.max() > 2.0:
            scale_factor = 255.0
            img_tensor = img_tensor / 255.0
        else:
            scale_factor = 1.0
    else:
        img_tensor = image.clone().float()
        original_shape = img_tensor.shape
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        elif img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
    
    # 按概率选择增强强度
    rand_val = random.random()
    
    if rand_val < 0.4:
        # 40% 不增强
        pass
    elif rand_val < 0.7:
        # 30% 弱增强
        img_tensor = apply_weak_augmentation(img_tensor)
    elif rand_val < 0.9:
        # 20% 中等增强
        img_tensor = apply_medium_augmentation(img_tensor)
    else:
        # 10% 强增强
        img_tensor = apply_strong_augmentation(img_tensor)
    
    # 转换回原格式
    if is_numpy:
        img_tensor = img_tensor * scale_factor
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
        if image.ndim == 2:
            img_np = img_np[:, :, 0]
        if img_dtype == np.uint8:
            img_np = img_np.clip(0, 255).astype(np.uint8)
        return img_np
    else:
        return img_tensor.view(original_shape)


# --- 对图像对应用域随机化（fix 和 moving 独立增强）---

def random_domain_augment_pair(fix_image, moving_image):
    """
    对图像对应用域随机化
    
    【关键】fix 和 moving 使用完全独立的随机参数和强度等级
    每张图像独立地按 4:3:2:1 的比例选择增强强度
    
    Args:
        fix_image: 固定图像 (Tensor [C, H, W] 或 [H, W])
        moving_image: 移动图像 (Tensor [C, H, W] 或 [H, W])
    
    Returns:
        fix_aug, moving_aug: 增强后的图像对
    """
    # 对两张图像分别独立应用域随机化（包括独立的强度选择）
    fix_aug = random_domain_augment_image(fix_image)
    moving_aug = random_domain_augment_image(moving_image)
    
    return fix_aug, moving_aug

