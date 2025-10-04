import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# =============================================================================
#  确保之前的函数 load_bin_file, SeismicShotDataset 等已经存在
# =============================================================================

def analyze_temporal_fft_distribution(all_shots_tensor, dt):
    """
    对一个包含N个样本的 [N, F, H, W] 张量进行时间上的FFT分解。
    计算并绘制整个数据集的平均时间频率功率谱。

    Args:
        all_shots_tensor (torch.Tensor or np.ndarray): 
            形状为 [N, F, H, W] 的数据张量。
            N: 样本数（炮数）
            F: 时间帧数 (Frames)
            H: 网格高度 (Height)
            W: 网格宽度 (Width)
        dt (float): 
            两个时间帧之间的时间间隔（单位：秒）。
            这是计算频率轴 (Hz) 的关键参数。
    """
    if dt <= 0:
        raise ValueError("时间采样间隔 'dt' 必须是一个正数。")

    # 确保输入是 numpy 数组以便进行后续处理
    if isinstance(all_shots_tensor, torch.Tensor):
        # .detach() 是一个好习惯，以防张量有关联的计算图
        data_np = all_shots_tensor.detach().cpu().numpy()
    else:
        data_np = np.asarray(all_shots_tensor)
        
    if data_np.ndim != 4:
        raise ValueError(f"输入张量的维度应为 4 (N, F, H, W)，但收到了 {data_np.ndim} 维。")

    num_shots, num_frames, height, width = data_np.shape
    print(f"\n开始对 {num_shots} 个样本的 [F, H, W] 数据进行时间FFT分析...")
    print(f"数据维度: N={num_shots}, F={num_frames}, H={height}, W={width}")
    print(f"时间采样间隔 dt = {dt} s")

    # 1. 计算频率轴（只需要计算一次）
    # np.fft.fftfreq 会生成频率坐标
    frequencies = np.fft.fftfreq(num_frames, d=dt)
    # 我们只关心正频率部分
    positive_freq_indices = np.where(frequencies >= 0)
    freq_axis_hz = frequencies[positive_freq_indices]

    # 2. 沿时间轴 (axis=1) 对所有样本的所有像素点一次性进行FFT
    # 这是向量化操作，效率远高于循环
    print("正在对整个数据集进行FFT计算...")
    fft_result = np.fft.fft(data_np, axis=1)
    
    # 3. 计算功率谱
    power_spectrum = np.abs(fft_result)**2
    
    # 4. 只取正频率部分
    # 我们需要在第二个维度（频率轴）上进行切片
    power_spectrum_positive = power_spectrum[:, positive_freq_indices[0], :, :]
    
    # 5. 对所有空间点 (H, W) 和所有炮 (N) 求平均，得到最终的平均功率谱
    # 我们需要对 axis 0, 2, 3 求平均
    print("FFT计算完成，正在计算最终平均谱并绘图...")
    final_mean_spectrum = np.mean(power_spectrum_positive, axis=(0, 2, 3))
    
    # --- 绘图 ---
    plt.figure(figsize=(12, 7))
    
    plt.plot(freq_axis_hz, final_mean_spectrum, color='green', linewidth=2)
    
    # 找到峰值频率
    peak_power_index = np.argmax(final_mean_spectrum)
    peak_frequency = freq_axis_hz[peak_power_index]
    peak_power = final_mean_spectrum[peak_power_index]
    
    plt.axvline(peak_frequency, color='red', linestyle='--', label=f'Peak Freq: {peak_frequency:.2f} Hz')
    plt.title('Average Temporal Frequency Power Spectrum (All Locations, All Shots)')
    plt.xlabel('Temporal Frequency (Hz)')
    plt.ylabel('Average Power (Log Scale)')
    plt.yscale('log') # 功率谱通常跨越多个数量级，对数尺度更合适
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    
    # 设置X轴范围，通常我们不关心非常高的频率
    # 奈奎斯特频率是 1 / (2 * dt)
    nyquist_freq = 1 / (2 * dt)
    plt.xlim(0, nyquist_freq) 
    
    print(f"分析完成。峰值频率出现在: {peak_frequency:.2f} Hz")

    plt.tight_layout()
    plt.savefig("频域.png")
    print("图像已保存为 '频域.png'")

