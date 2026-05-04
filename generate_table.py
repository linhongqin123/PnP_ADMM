import torch
import torch.nn as nn
import numpy as np
import warnings
# 严格按照作业要求导入指标计算函数
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from skimage import data
from skimage.color import rgb2gray

warnings.filterwarnings("ignore")

# ==========================================
# 1. 网络架构定义 (DnCNN)
# ==========================================
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
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
# 2. PnP-ADMM 核心算法 (自然图像去噪)
# ==========================================
def pnp_admm_denoise(y, model, rho=0.5, max_iter=15):
    x = y.clone()
    v = y.clone()
    u = torch.zeros_like(y)
    
    for k in range(max_iter):
        x = (y + rho * (v - u)) / (1 + rho)
        v = denoise(model, x + u)
        u = u + x - v
    return x

# ==========================================
# 3. 辅助评估函数
# ==========================================
def calculate_psnr_ssim(img_true, img_test):
    p = psnr_metric(img_true, img_test, data_range=1.0)
    s = ssim_metric(img_true, img_test, data_range=1.0)
    return p, s

# ==========================================
# 4. 批量评测与表格生成 (移除 Pandas 版本)
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device} 进行批量评估...\n")

    # 加载模型
    model = DnCNN().to(device)
    try:
        loaded_data = torch.load('DnCNN.pth', map_location=device, weights_only=False)
        if isinstance(loaded_data, nn.Module):
            model = loaded_data.to(device)
        else:
            new_state_dict = {k.replace("module.", ""): v for k, v in loaded_data.items()}
            model.load_state_dict(new_state_dict)
        print("✅ DnCNN 模型加载成功！开始生成数据表...\n")
    except Exception as e:
        print(f"⚠️ 无法加载 DnCNN.pth ({e})。使用随机权重运行演示。\n")

    test_images = {
        'Camera': data.camera() / 255.0,
        'Chelsea': rgb2gray(data.chelsea())
    }
    noise_levels = [15, 25, 50]
    results = []

    for img_name, img_np in test_images.items():
        img_np = img_np.astype(np.float32)
        x_true_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
        
        for sigma in noise_levels:
            noise_std = sigma / 255.0
            torch.manual_seed(42)
            noise = torch.randn_like(x_true_tensor) * noise_std
            y_noisy = (x_true_tensor + noise).clamp(0, 1)
            
            current_rho = 0.5 if sigma <= 25 else 0.8 
            x_recon = pnp_admm_denoise(y_noisy, model, rho=current_rho, max_iter=20)
            
            x_true_np = x_true_tensor.squeeze().cpu().numpy()
            y_noisy_np = y_noisy.squeeze().cpu().numpy()
            x_recon_np = x_recon.squeeze().cpu().numpy()
            
            p_noisy, s_noisy = calculate_psnr_ssim(x_true_np, y_noisy_np)
            p_recon, s_recon = calculate_psnr_ssim(x_true_np, x_recon_np)
            
            results.append([
                img_name, 
                str(sigma), 
                f"{p_noisy:.2f}", f"{s_noisy:.4f}", 
                f"{p_recon:.2f}", f"{s_recon:.4f}", 
                f"+{p_recon - p_noisy:.2f}"
            ])

    # 纯手工打印 Markdown 表格
    print("-" * 85)
    print("定量对比结果表格：加噪图像 vs PnP-ADMM 恢复图像")
    print("-" * 85)
    print("| Image | Noise Level (σ) | Noisy PSNR | Noisy SSIM | PnP-ADMM PSNR | PnP-ADMM SSIM | PSNR Gain |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in results:
        print(f"| {row[0]:^5} | {row[1]:^15} | {row[2]:^10} | {row[3]:^10} | {row[4]:^13} | {row[5]:^13} | {row[6]:^9} |")
    print("-" * 85)
    print("\n你可以直接复制上方表格粘贴到你的实验报告中！")