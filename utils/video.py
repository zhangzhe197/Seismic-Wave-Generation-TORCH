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
# --- 全局配置 ---
# FFMPEG 可执行文件的路径，如果不在系统 PATH 中，请指定完整路径
FFMPEG_EXE = "ffmpeg" 

# --- 视频创建函数 (源自你的参考代码，功能强大，直接复用) ---
def create_video_from_frames(input_image_pattern, output_video_path, fps=10, cleanup=True, temp_dir=None):
    """
    使用 FFmpeg 从一系列图片帧创建视频。
    
    Args:
        input_image_pattern (str): 输入图片的格式化路径 (例如, 'frames/frame_%04d.png').
        output_video_path (str): 输出视频的完整路径.
        fps (int): 视频的帧率.
        cleanup (bool): 是否在视频创建后删除图片帧和临时目录.
        temp_dir (str, optional): 存放图片帧的临时目录路径，用于清理.
    """
    print(f"\n尝试使用 FFmpeg 创建视频: {output_video_path} ...")
    command = [
        FFMPEG_EXE, "-y", "-framerate", str(fps), "-i", input_image_pattern,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-vf", f"fps={fps},format=yuv420p", "-movflags", "+faststart",
        output_video_path
    ]
    try:
        # 运行ffmpeg命令，隐藏其输出，但在出错时捕获
        subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"视频 '{output_video_path}' 创建成功！")
        
        if cleanup:
            print("正在清理临时文件...")
            # 删除图片帧
            for img_file in glob.glob(input_image_pattern.replace('%04d', '*')):
                try: 
                    os.remove(img_file)
                except OSError as e: 
                    print(f"删除文件 {img_file} 失败: {e}")
            
            # 删除临时目录（如果它是空的）
            if temp_dir:
                try:
                    if not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                except OSError as e:
                    print(f"删除临时目录 {temp_dir} 失败: {e}")
            print("临时文件清理完成。")
            
    except FileNotFoundError:
        print(f"错误: 找不到 FFmpeg 可执行文件 '{FFMPEG_EXE}'。请确保 FFmpeg 已安装并添加到系统 PATH 中。")
    except subprocess.CalledProcessError as e:
        print(f"错误: FFmpeg 命令执行失败 (返回码: {e.returncode})。\nFFmpeg stderr:\n{e.stderr}")


# --- 核心函数：从 NumPy 数组创建视频 ---
def create_video_from_numpy(data_array, output_video_path, fps=25, title_prefix="", cleanup_frames=True):
    """
    将一个3D NumPy数组 (frames, height, width) 转换为视频。

    Args:
        data_array (np.ndarray): 输入的3D NumPy数组。
        output_video_path (str): 输出视频的路径 (例如, './output/my_video.mp4').
        fps (int): 生成视频的帧率。
        title_prefix (str): 每帧图像标题的前缀。
        cleanup_frames (bool): 是否在完成后删除临时图片。
    """
    # 1. 输入验证
    if not isinstance(data_array, np.ndarray) or data_array.ndim != 3:
        raise ValueError("输入必须是一个3维的 NumPy 数组 (frames, height, width)。")
    
    num_frames, height, width = data_array.shape
    print(f"检测到输入数据: {num_frames} 帧, 每帧尺寸 {height}x{width}.")

    # 2. 创建临时目录存放图片帧
    output_dir = os.path.dirname(output_video_path)
    temp_frame_dir = os.path.join(output_dir, "temp_frames_for_video")
    os.makedirs(temp_frame_dir, exist_ok=True)
    print(f"临时图片将保存在: {temp_frame_dir}")

    # 3. 预扫描数据以确定全局颜色范围，确保颜色条一致
    print("正在确定全局颜色范围...")
    global_min = data_array.min()
    global_max = data_array.max()
    global_abs_max = max(abs(global_min), abs(global_max))

    if global_abs_max == 0:
        print("警告: 所有数据值均为0。")
        global_abs_max = 1.0 # 避免除以零

    # 使用对称对数归一化 (SymLogNorm)，非常适合中心在0附近且有极端值的数据
    linthresh = global_abs_max / 100.0
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-global_abs_max, vmax=global_abs_max, base=10)
    
    print(f"全局数据范围: [{global_min:.4g}, {global_max:.4g}]")
    print(f"颜色条范围 (SymLogNorm): [{-global_abs_max:.4g}, {global_abs_max:.4g}], 线性阈值: {linthresh:.4g}")

    # 4. 逐帧生成并保存图片
    print("正在生成图片帧...")
    for i in tqdm(range(num_frames), desc="生成帧"):
        frame_data = data_array[i, :, :]
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        im = ax.imshow(frame_data, cmap='RdBu_r', origin='upper', norm=norm)
        fig.colorbar(im, ax=ax, label="Amplitude")
        
        title = f"{title_prefix} - Frame {i+1}/{num_frames}"
        ax.set_title(title)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        
        fig.tight_layout()
        
        image_filename = os.path.join(temp_frame_dir, f"frame_{i:05d}.png")
        plt.savefig(image_filename, dpi=100) # dpi可以控制图片清晰度
        plt.close(fig) # **极其重要**：关闭图形以释放内存

    # 5. 调用FFmpeg合成视频
    input_pattern = os.path.join(temp_frame_dir, "frame_%05d.png")
    create_video_from_frames(
        input_pattern, 
        output_video_path, 
        fps=fps, 
        cleanup=cleanup_frames, 
        temp_dir=temp_frame_dir
    )

