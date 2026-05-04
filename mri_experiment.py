import torch
import torch.nn as nn
import numpy as np
import warnings
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from skimage import data
from skimage.color import rgb2gray

warnings.filterwarnings("ignore")

# ==========================================
# 1. 核心网络 (DnCNN)
# ==========================================
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True):
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)

def denoise(model, v_u):
    model.eval()
    with torch.no_grad():
        res = model(v_u)
    return res.clamp(0, 1)

# ==========================================
# 2. PnP-ADMM MRI 频域重建核心算法
# ==========================================
def pnp_admm_mri(y_kspace, mask, model, rho=0.05, max_iter=30):
    """
    y_kspace: 带 Mask 的频域观测数据 (K-space)
    mask: 采样掩码 (0/1)
    """
    # 初始化 x 为零填充(Zero-filling)的逆傅里叶变换
    x = torch.fft.ifft2(y_kspace).real
    v = x.clone()
    u = torch.zeros_like(x)
    
    for k in range(max_iter):
        # 1. 数据保真步 (x-update) - 频域闭式解
        term1 = y_kspace
        term2 = rho * torch.fft.fft2(v - u)
        x_kspace = (term1 + term2) / (mask + rho)
        x = torch.fft.ifft2(x_kspace).real
        
        # 2. 先验去噪步 (v-update) - 图像域去噪
        v = denoise(model, x + u)
        
        # 3. 对偶变量更新 (u-update)
        u = u + x - v
        
    return x.clamp(0, 1)

# ==========================================
# 3. 辅助函数：生成 MRI 欠采样 Mask
# ==========================================
def generate_2d_mask(shape, sampling_ratio=0.2):
    """生成保留中心低频，随机采样外围高频的 MRI 采样掩码"""
    mask = np.zeros(shape, dtype=np.float32)
    h, w = shape
    center_h, center_w = h // 2, w // 2
    # 固定保留中心 8% 的极低频区域 (MRI 能量集中于此)
    lh, lw = int(h * 0.04), int(w * 0.04)
    mask[center_h - lh:center_h + lh, center_w - lw:center_w + lw] = 1
    
    # 在其余区域随机采样，凑够总的 sampling_ratio
    num_samples = int(h * w * sampling_ratio) - (2*lh * 2*lw)
    if num_samples > 0:
        flat_mask = mask.flatten()
        zero_indices = np.where(flat_mask == 0)[0]
        random_indices = np.random.choice(zero_indices, num_samples, replace=False)
        flat_mask[random_indices] = 1
        mask = flat_mask.reshape(shape)
    return mask

# ==========================================
# 4. 主评测流程
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 启动任务 B (MRI重建) 评测，使用设备: {device}...\n")

    # 1. 加载模型
    model = DnCNN().to(device)
    try:
        loaded_data = torch.load('DnCNN.pth', map_location=device, weights_only=False)
        if isinstance(loaded_data, nn.Module):
            model = loaded_data.to(device)
        else:
            new_state_dict = {k.replace("module.", ""): v for k, v in loaded_data.items()}
            model.load_state_dict(new_state_dict)
        print("✅ DnCNN 模型加载成功！\n")
    except:
        print("⚠️ 无法加载模型，使用未初始化权重演示。\n")

    # 2. 准备图像 (模拟医学图像，转为 256x256 灰度图)
    # 注：如果你有真实的核磁共振切片，可以替换此处
    img_np = data.camera()
    import cv2
    img_np = cv2.resize(img_np, (256, 256))
    img_np = np.array(img_np, dtype=np.float32) / 255.0
    x_true_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    
    sampling_ratios = [0.1, 0.2, 0.3] # 对应 10%, 20%, 30% 采样率
    results = []
    
    # 准备可视化画布
    fig, axes = plt.subplots(len(sampling_ratios), 3, figsize=(10, 3 * len(sampling_ratios)))
    fig.suptitle('Task B: PnP-ADMM MRI Reconstruction', fontsize=16)

    # 3. 循环测试不同采样率
    for idx, ratio in enumerate(sampling_ratios):
        # 生成掩码并转换到 K 空间
        torch.manual_seed(42)
        np.random.seed(42)
        mask_np = generate_2d_mask(img_np.shape, sampling_ratio=ratio)

        mask_np = np.fft.ifftshift(mask_np)
        
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
        
        # 物理退化：F(x) * Mask
        x_complex = x_true_tensor + 0j
        k_space_full = torch.fft.fft2(x_complex)
        y_kspace = mask * k_space_full
        
        # 基线方法：零填充直接逆变换 (Zero-Filling)
        zf_recon = torch.fft.ifft2(y_kspace).real.clamp(0, 1)
        
        # PnP-ADMM 重建
        pnp_recon = pnp_admm_mri(y_kspace, mask, model, rho=0.05, max_iter=30)
        
        # 计算指标
        x_true_numpy = img_np
        zf_numpy = zf_recon.squeeze().cpu().numpy()
        pnp_numpy = pnp_recon.squeeze().cpu().numpy()
        
        p_zf, s_zf = psnr_metric(x_true_numpy, zf_numpy, data_range=1.0), ssim_metric(x_true_numpy, zf_numpy, data_range=1.0)
        p_pnp, s_pnp = psnr_metric(x_true_numpy, pnp_numpy, data_range=1.0), ssim_metric(x_true_numpy, pnp_numpy, data_range=1.0)
        
        results.append([f"{ratio*100:.0f}%", f"{p_zf:.2f}", f"{s_zf:.4f}", f"{p_pnp:.2f}", f"{s_pnp:.4f}", f"+{p_pnp - p_zf:.2f}"])
        
        # 可视化填充
        ax_gt, ax_zf, ax_pnp = axes[idx]
        ax_gt.imshow(x_true_numpy, cmap='gray'); ax_gt.set_title("Ground Truth"); ax_gt.axis('off')
        ax_zf.imshow(zf_numpy, cmap='gray'); ax_zf.set_title(f"Zero-Filling (SR={ratio*100:.0f}%)"); ax_zf.axis('off')
        ax_pnp.imshow(pnp_numpy, cmap='gray'); ax_pnp.set_title(f"PnP-ADMM (PSNR:{p_pnp:.2f})"); ax_pnp.axis('off')

    # 4. 打印 Markdown 表格
    print("-" * 80)
    print("任务 B：MRI 重建定量对比结果 (Zero-Filling vs PnP-ADMM)")
    print("-" * 80)
    print("| Sampling Rate | Zero-Fill PSNR | Zero-Fill SSIM | PnP-ADMM PSNR | PnP-ADMM SSIM | PSNR Gain |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(f"| {r[0]:^13} | {r[1]:^14} | {r[2]:^14} | {r[3]:^13} | {r[4]:^13} | {r[5]:^9} |")
    print("-" * 80)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('MRI_Reconstruction_Results.png', dpi=300)
    print("\n📸 对比可视化图像已自动保存为当前目录下的 'MRI_Reconstruction_Results.png'！")