

***

# Plug-and-Play ADMM 图像恢复

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

本项目复现了经典论文 ["Plug-and-Play ADMM for Image Restoration: Fixed Point Convergence and Applications"](https://arxiv.org/abs/1605.01710) 中的核心算法，通过交替方向乘子法（ADMM）框架，将预训练的深度学习去噪模型（DnCNN）作为先验（Prior）无缝嵌入到图像恢复任务中。

##  项目简介

传统的基于物理模型的优化方法（如 TV-ADMM）通常受限于手工设计先验的表达能力，而端到端的深度学习方法又缺乏在不同逆问题间的通用性。Plug-and-Play (PnP) ADMM 框架巧妙地解决了这一矛盾。它将“数据保真项”与“先验项”解耦，使我们能够直接利用现成的深度去噪器作为隐式的神经网络先验，**无需针对特定任务重新训练网络**。

本项目成功将 PnP-ADMM 框架应用于两大经典的计算成像任务：
- **Task A：自然图像去噪**（去除高斯噪声）
- **Task B：MRI 图像重建**（基于 K 空间欠采样的频域重建）

##  目录结构

```text
├── main.py                  # 包含 Task A 和 Task B 的端到端执行脚本
├── generate_table.py        # Task A 的批量评估脚本（用于生成定量结果表格）
├── mri_experiment.py        # 专用于 Task B（K空间欠采样重建）的独立测试脚本
├── DnCNN.pth                # 作为 PnP 先验使用的预训练 DnCNN 模型权重
├── Figure_1.png             # 自然图像去噪任务的可视化对比图
├── Figure_2.png             # MRI 重建任务的多重采样率可视化对比图
├── report-final.pdf         # 最终的学术实验报告（由 LaTeX 编译生成）
└── README.md                # 项目说明文档
```

##  环境依赖与安装

请确保你的运行环境安装了 Python 3.8+。本项目的矩阵计算经过了针对 GPU 的深度优化（推荐使用 NVIDIA RTX 系列以获取最佳性能）。请使用 `pip` 安装以下依赖包：

```bash
pip install torch torchvision
pip install numpy pandas matplotlib scikit-image opencv-python
```

##  运行指南

### 1. 自然图像去噪 (Task A)
评估 PnP-ADMM 在不同高斯噪声水平（$\sigma = 15, 25, 50$）下的去噪性能，并自动生成定量对比表格：
```bash
python generate_table.py
```

### 2. MRI 图像重建 (Task B)
模拟 MRI 扫描过程中的 K 空间欠采样（保留 10%、20%、30% 的低频及随机数据），并对比传统的零填充法（Zero-Filling）与 PnP-ADMM 的重建效果：
```bash
python mri_experiment.py
```
*技术细节注记：在实现过程中，我们特别使用了 `np.fft.ifftshift` 对生成的掩码进行零频移位对齐，以迎合 PyTorch FFT 算法默认将低频分量置于矩阵四角的特性。这一关键修复成功将重建的 PSNR 修正至正常的 30dB+ 水平。*

##  核心实验结果

实验证明，PnP-ADMM 框架展现出了极强的鲁棒性与泛化能力：
* **去噪任务 (Denoising)**：在重度噪声污染下，相较于带噪原图，实现了 **+7dB 至 +11dB** 的显著 PSNR 提升。
* **MRI 重建 (MRI Reconstruction)**：即使在极端截断的频域数据下，依然能锐利地恢复出高频边缘细节（例如：在 30% 采样率下，相较于基线方法取得了 +8.30dB 的 PSNR 增益）。

```
