import torch
import numpy as np
import matplotlib.pyplot as plt
# 导入中文显示库
import matplotlib
import time, os,pdb
import torch.nn as nn


import os
import subprocess
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm # 引入 tqdm 用于显示进度条
from utils.freqAna import analyze_temporal_fft_distribution
from utils.video import create_video_from_numpy
from utils.freqVis import vis_freq
# --- 全局配置 ---
# FFMPEG 可执行文件的路径，如果不在系统 PATH 中，请指定完整路径


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pml_width = 50
pml_amp = 1000.0
dz, dx = 10.0, 10.0
target_width = 175
nt = 1000
dt = 0.001
f0 = 15

res_width = 70


raw_target_point = [[31, 0], [41, 0] , [51, 0], [21,0]]
target_Source_Point = [[int(point[0] * 25 / dz + 0.5), 0] for point in raw_target_point]



t_vec = torch.arange(0, nt * dt, dt, device=device)



nz, nx = target_width + pml_width * 2, target_width + pml_width * 2








v_ROOT_DIR = "/home/niuma009/HDD/petrel/zcc/CA/model"


def ricker_wavelet(f, t):
    """ Ricker wavelet generator """
    t0 = 1.0 / f
    t_shifted = t - t0
    arg = (np.pi * f * t_shifted)**2
    wavelet = (1 - 2 * arg) * np.exp(-arg)
    return torch.tensor(wavelet, dtype=torch.float32, device=device)

def create_pml_profiles(nz, nx, pml_width, pml_amp, dt, v):
    """ 创建PML阻尼剖面并计算正确的更新系数 """
    sigma_z = torch.zeros(nz, nx, device=device)
    sigma_x = torch.zeros(nz, nx, device=device)
    # z方向的PML
    for i in range(pml_width):
        val = pml_amp * ((pml_width - i) / pml_width) ** 2
        sigma_z[i, :] = val
        sigma_z[nz - 1 - i, :] = val
    # x方向的PML
    for i in range(pml_width):
        val = pml_amp * ((pml_width - i) / pml_width) ** 2
        sigma_x[:, i] = val
        sigma_x[:, nx - 1 - i] = val
    ax = (sigma_x * dt) / 2.0
    az = (sigma_z * dt) / 2.0
    c1x = (2) / (1 + ax)
    c2x = (1 - ax) / (1 + ax)
    c3x = (v**2 * dt**2) / (1 + ax)
    c1z = (2) / (1 + az)
    c2z = (1 - az) / (1 + az)
    c3z = (v**2 * dt**2) / (1 + az)
    return c1x, c2x, c3x, c1z, c2z, c3z


def wave_sim(v_in : np.ndarray, src_z, src_x):

    v = torch.from_numpy(v_in).to(device).unsqueeze(0)
    v = nn.functional.interpolate(v, size=( target_width, target_width  ), mode="nearest")
    v = nn.functional.pad(v, (pml_width, pml_width , pml_width, pml_width), "replicate")
    v = v.view(nx,nz)
    source = ricker_wavelet(f0, t_vec.cpu().numpy())

    # 使用修正后的函数获取系数
    c1x, c2x, c3x, c1z, c2z, c3z = create_pml_profiles(nz, nx, pml_width, pml_amp, dt, v)

    # 初始化波场变量
    p = torch.zeros(nz, nx, device=device)
    px = torch.zeros(nz, nx, device=device)
    pz = torch.zeros(nz, nx, device=device)
    p_prev = torch.zeros(nz, nx, device=device)
    px_prev = torch.zeros(nz, nx, device=device)
    pz_prev = torch.zeros(nz, nx, device=device)

    # <<< 修改部分 1: 定义六阶精度有限差分卷积核 >>>
    # 六阶精度系数: [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90] / (dx^2)
    c_6th = torch.tensor([1/90.0, -3/20.0, 3/2.0, -49/18.0, 3/2.0, -3/20.0, 1/90.0], dtype=torch.float32, device=device)

    kernel_x_6th = c_6th.reshape(1, 1, 1, 7) / (dx**2)
    kernel_z_6th = c_6th.reshape(1, 1, 7, 1) / (dz**2)
    # <<< 修改结束 >>>


    snapshots = []

    # --- 4. 时间步进循环 ---
    for it in range(nt):
        p_batch = p.unsqueeze(0).unsqueeze(0)

        # <<< 修改部分 2: 使用六阶核并更新padding >>>
        # 六阶核尺寸为7，需要padding=3以保持尺寸不变
        lap_x = torch.nn.functional.conv2d(p_batch, kernel_x_6th, padding=(0, 3)).squeeze()
        lap_z = torch.nn.functional.conv2d(p_batch, kernel_z_6th, padding=(3, 0)).squeeze()
        # <<< 修改结束 >>>

        px_new = c1x * px - c2x * px_prev + c3x * lap_x
        pz_new = c1z * pz - c2z * pz_prev + c3z * lap_z
        
        p_new = px_new + pz_new
        p_new[src_z, src_x] += source[it] 
        
        px_prev, px = px.clone(), px_new
        pz_prev, pz = pz.clone(), pz_new
        p = p_new
        

        snapshots.append(p.detach().clone()[pml_width:target_width + pml_width, pml_width : pml_width + target_width])
        
    res = torch.stack(snapshots, dim=0).cpu()
    # res = nn.functional.interpolate(res.unsqueeze(1) , size=( res_width, res_width  )).view(-1, res_width ,res_width)
    return res.numpy()



def main():
    """
    遍历所有速度模型和炮点，对每个波场模拟结果进行傅里叶变换，
    并保存最低的50个频率分量到 .npy 文件中。
    """
    # 1. 定义并创建用于存储频域数据的新目录
    v_OUTPUT_DIR = "/home/niuma009/HDD/newGenData/"
    os.makedirs(v_OUTPUT_DIR, exist_ok=True)
    print(f"频域数据将保存至: '{v_OUTPUT_DIR}/'")

    # 获取所有速度模型文件
    files = sorted(os.listdir(v_ROOT_DIR)) # 使用 sorted() 保证处理顺序
    res_freq = []
    for file in files[:20]:
        # 注意：这里假设文件名可以唯一标识一个速度模型。
        # 如果需要从文件名中提取索引，可以这样做：
        # model_index = int(file.split('_')[-1].split('.')[0])
        # 这里我们为了简化，直接使用循环的 i
        
        v_field_numpy = np.load(os.path.join(v_ROOT_DIR, file))
       

        # 遍历速度模型集合中的每一个模型
        for i in tqdm(range(v_field_numpy.shape[0])): # 假设v_field_numpy的第一个维度是模型的数量            
            # 使用 enumerate 获取炮点的索引，方便命名文件
            rrmse_results = []
            for j, point in enumerate(target_Source_Point):
                            # 定义您想要用来重建波场的特定频率 (单位: Hz)
                    target_frequencies_hz = [i for i in range(0, 40, 1)]
                    # 用于存储当前速度模型下，不同炮点的RRMSE结果
                    st = time.time()
                    wave_snapshot = wave_sim(v_field_numpy[i], point[1] + pml_width, point[0] + pml_width)[::1, : ,: ]
                    print(time.time() - st)
                    vis_freq(torch.from_numpy(wave_snapshot))
                    # create_video_from_numpy(wave_snapshot, "newVideo.mp4", 10)
                    # res_freq.append(torch.from_numpy(wave_snapshot))
                    # if len(res_freq) > 100:
                    #     analyze_temporal_fft_distribution(torch.stack(res_freq, dim=0), dt=1e-2)
                    #     pdb.set_trace()

                    # 2. 沿着时间轴进行快速傅里叶变换
                    fft_wave = np.fft.rfft(wave_snapshot, axis=0)
                    pdb.set_trace()
                    # 3. 确定每个FFT索引对应的实际频率 (Hz)
                    freq_axis_hz = np.fft.rfftfreq(nt, d=dt)

                    # 4. 找到与目标频率最接近的FFT索引
                    target_indices = []
                    for f_hz in target_frequencies_hz:
                        idx = np.argmin(np.abs(freq_axis_hz - f_hz))
                        target_indices.append(idx)
                    target_indices = sorted(list(set(target_indices)))
                    
                    print(f"\n处理中... 速度模型: {i}, 炮点: {point}")
                    print(f"  目标频率 (Hz): {target_frequencies_hz}")
                    print(f"  映射到的FFT索引: {target_indices}")
                    # 5. 创建一个只包含目标频率分量的新的频域数组
                    filtered_fft_wave = np.zeros_like(fft_wave)
                    filtered_fft_wave[target_indices, :, :] = fft_wave[target_indices, :, :]
                    # 6. 沿着时间轴进行逆快速傅里叶变换以重建波场
                    reconstructed_snapshot = np.fft.irfft(filtered_fft_wave, n=nt, axis=0)

                    # 7. 计算相对均方根误差 (RRMSE)
                    # 7.1 计算误差的均方值 (MSE)
                    mse = np.mean((wave_snapshot - reconstructed_snapshot)**2)
                    
                    # 7.2 计算原始信号的均方值
                    signal_power = np.mean(wave_snapshot**2)
                    
                    # 7.3 计算RRMSE
                    # 添加一个极小值 epsilon 以防止 signal_power 为零时出现除零错误
                    epsilon = 1e-12
                    rrmse = (mse / (signal_power + epsilon))
                    rrmse_results.append(rrmse)
                    
                    # RRMSE通常以百分比形式表示，更直观
                    print(f"  计算得到的 RMSE: {rrmse:.6f} (或 {rrmse*100:.2f}%)")
                
                
                # (可选) 在处理完一个速度模型的所有炮点后，可以计算平均RRMSE
                    avg_rrmse = np.mean(rrmse_results)
                    print(f"\n速度模型 {i} 完成. 平均 RRMSE: {avg_rrmse:.6f} (或 {avg_rrmse*100:.2f}%)\n" + "="*50)
                    # break
                #     # 如果您仍然希望在每次循环后进行调试，可以保留 pdb.set_trace()
               
                

            
            # 如果您仍然希望在每次处理完一个速度模型后进行调试，可以保留
            pdb.set_trace()
if __name__ == "__main__":
    main()