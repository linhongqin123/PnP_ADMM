import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
        # DnCNN 预测的是噪声残差
        noise = self.dncnn(x)
        return x - noise 

def denoise(model, v_u):
    """封装去噪器调用"""
    model.eval()
    with torch.no_grad():
        res = model(v_u)
    return res.clamp(0, 1) # 将输出限制在合法像素范围内

# ==========================================
# 2. PnP-ADMM 核心算法
# ==========================================
def pnp_admm_denoise(y, model, rho=0.1, max_iter=20):
    """
    任务 A: 自然图像去噪
    y: 观测到的带噪图像 [1, 1, H, W]
    """
    x = y.clone()
    v = y.clone()
    u = torch.zeros_like(y)
    
    for k in range(max_iter):
        # 1. Data fidelity (x-update) - 闭式解
        x = (y + rho * (v - u)) / (1 + rho)
        
        # 2. Prior (v-update) - 调用 DnCNN
        v = denoise(model, x + u)
        
        # 3. Multiplier (u-update)
        u = u + x - v
        
    return x

def pnp_admm_mri(y_kspace, mask, model, rho=0.1, max_iter=20):
    """
    任务 B: MRI 重建
    y_kspace: 欠采样的 K 空间数据 (复数 Tensor)
    mask: 采样掩码
    """
    # 零填充傅里叶逆变换作为初始值 x
    x = torch.fft.ifft2(y_kspace).real
    v = x.clone()
    u = torch.zeros_like(x)
    
    for k in range(max_iter):
        # 1. Data fidelity (x-update) - 在频域求解
        # x = F^-1 ( (F(y) + rho * F(v - u)) / (Mask + rho) )
        term1 = y_kspace
        term2 = rho * torch.fft.fft2(v - u)
        x_kspace = (term1 + term2) / (mask + rho)
        x = torch.fft.ifft2(x_kspace).real
        
        # 2. Prior (v-update) - 调用 DnCNN 进行图像域去噪
        v = denoise(model, x + u)
        
        # 3. Multiplier (u-update)
        u = u + x - v
        
    return x

# ==========================================
# 3. 辅助函数 (评估与可视化)
# ==========================================
def calculate_metrics(img_true, img_test):
    true_np = img_true.squeeze().cpu().numpy()
    test_np = img_test.squeeze().cpu().numpy()
    p = psnr(true_np, test_np, data_range=1.0)
    s = ssim(true_np, test_np, data_range=1.0)
    return p, s

def show_results(title, img_clean, img_degraded, img_recon, psnr_val, ssim_val):
    plt.figure(figsize=(12, 4))
    plt.suptitle(title, fontsize=16)
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_clean.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_degraded.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Degraded (Input)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_recon.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f"PnP-ADMM\nPSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.4f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. 主执行流
# ==========================================
if __name__ == "__main__":
    # 配置设备：发挥你 RTX 5070 的威力！
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

   # 1. 加载模型
    model = DnCNN().to(device)
    try:
        # 针对本地信任文件，显式关闭 weights_only 限制
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning) # 屏蔽烦人的未来版本警告
        
        loaded_data = torch.load('DnCNN.pth', map_location=device, weights_only=False)
        
        # 兼容性处理：判断加载出来的是整个网络对象，还是纯权重字典
        if isinstance(loaded_data, nn.Module):
            model = loaded_data.to(device)
            print("✅ 检测到保存的是完整模型对象，DnCNN 加载成功！")
        else:
            # 如果存在多卡训练的前缀 'module.'，自动剥离它
            new_state_dict = {}
            for k, v in loaded_data.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print("✅ DnCNN 模型权重 (state_dict) 加载成功！")
            
    except Exception as e:
        print(f"❌ 警告：无法加载 DnCNN.pth ({e})。")

    # 2. 读取并准备图像数据
    # 请确保有一张图片，这里将其统一调整为 256x256，并归一化到 [0,1]
    img = cv2.imread('test_image.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("未找到 test_image.png，使用合成测试图...")
        img = np.random.rand(256, 256).astype(np.float32)
    else:
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
    
    x_true = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    print("\n--- 启动任务 A: 自然图像去噪 ---")
    # 生成带噪图像 (加入 sigma=25/255 的高斯白噪声)
    noise_level = 25.0 / 255.0
    noise = torch.randn_like(x_true) * noise_level
    y_noisy = x_true + noise

    # 运行 PnP-ADMM 去噪
    rho_denoise = 0.5
    iterations = 15
    x_recon_denoise = pnp_admm_denoise(y_noisy, model, rho=rho_denoise, max_iter=iterations)
    
    # 评估与可视化
    p_denoise, s_denoise = calculate_metrics(x_true, x_recon_denoise)
    show_results("Task A: Natural Image Denoising", x_true, y_noisy.clamp(0,1), x_recon_denoise, p_denoise, s_denoise)


    print("\n--- 启动任务 B: MRI 重建 ---")
    # 生成 1D 随机高斯采样掩码 (模拟 MRI 欠采样，加速比约 3-4 倍)
    mask_np = np.zeros((256, 256), dtype=np.float32)
    center = 256 // 2
    # 保留低频中心区域
    mask_np[:, center-10:center+10] = 1 
    # 随机保留高频区域
    mask_np[:, np.random.choice(256, 50, replace=False)] = 1 
    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

    # 生成 MRI 观测数据 y = Mask * F(x)
    x_complex = x_true + 0j # 转为复数以便进行傅里叶变换
    k_space_full = torch.fft.fft2(x_complex)
    y_kspace = mask * k_space_full
    
    # 零填充（Zero-filling）的直接逆变换结果，用作对比基线
    zero_filled_recon = torch.fft.ifft2(y_kspace).real

    # 运行 PnP-ADMM MRI 重建
    rho_mri = 0.05
    iterations_mri = 30
    x_recon_mri = pnp_admm_mri(y_kspace, mask, model, rho=rho_mri, max_iter=iterations_mri)

    # 评估与可视化
    p_mri, s_mri = calculate_metrics(x_true, x_recon_mri)
    show_results("Task B: MRI Reconstruction (K-space Sub-sampling)", x_true, zero_filled_recon.clamp(0,1), x_recon_mri, p_mri, s_mri)
    
    print("\n🎉 实验全部运行完毕！请保存图表并整理实验报告。")