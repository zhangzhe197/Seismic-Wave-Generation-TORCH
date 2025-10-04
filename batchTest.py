import torch, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time, os, pdb
import torch.nn as nn
from tqdm import tqdm
# from utils.video import create_video_from_numpy
# from utils.drawVel import plot_array_with_highlight
# --- 1. 全局配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的计算设备: {device}")

# 物理和模拟参数

pml_width = 50                      # PML边界层宽度  
pml_amp = 1000.0                    # PML衰减参数
dz, dx = 14.0, 14.0                 # z, x 上的网格大小, 单位m
target_width = 125                  # 实际模拟的时候的网格大小, 这里模拟125 x 125的网格, 实际上是1750mx1750m的模拟场
nt = 1000                           # 要模拟多少个时间步
dt = 0.001                          # 时间步长大小 单位s
f0 = 8.0                            # 生成Ricker子波的频率 hz
res_width = 70                      # 输出结果的尺寸, 输出的结果将会降采样到这个大小
test_sample_num = 3000              # 划分数据集的时候, 测试集的速度场数量
numFromAClass = 15000               # 一个类别的总共的速度场的数量
BATCH_SIZE = 32                     # 批处理计算的大小
# 震源点定义
raw_target_point = [[21, 0], [31, 0], [41, 0], [51, 0]]
target_Source_Point = [[int(point[0] * 25 / dz + 0.5) + pml_width, pml_width] for point in raw_target_point]

# 时间向量和计算域尺寸
t_vec = torch.arange(0, nt * dt, dt, device=device)
nz, nx = target_width + pml_width * 2, target_width + pml_width * 2

# 数据路径
v_ROOT_DIR = ["/home/niuma009/HDD/petrel/zcc/CA/model", "/home/niuma009/HDD/petrel/zcc/style-A/model", "/home/niuma009/HDD/petrel/zcc/FFB/model"]
v_OUTPUT_DIR = "/home/niuma009/data/newGenData/"




def ricker_wavelet(f, t):
    t0 = 1.0 / f
    t_shifted = t - t0
    arg = (np.pi * f * t_shifted)**2
    wavelet = (1 - 2 * arg) * np.exp(-arg)
    return torch.tensor(wavelet, dtype=torch.float32, device=device)

def create_pml_profiles_batched(batch_size, nz, nx, pml_width, pml_amp, dt, v_batch):

    sigma_z_base = torch.zeros(nz, nx, device=device)
    sigma_x_base = torch.zeros(nz, nx, device=device)
    
    for i in range(pml_width):
        val = pml_amp * ((pml_width - i) / pml_width) ** 2
        sigma_z_base[i, :] = val
        sigma_z_base[nz - 1 - i, :] = val
        sigma_x_base[:, i] = val
        sigma_x_base[:, nx - 1 - i] = val

    # (nz, nx) -> (1, nz, nx) -> (batch_size, nz, nx)
    sigma_z = sigma_z_base.unsqueeze(0).expand(batch_size, -1, -1)
    sigma_x = sigma_x_base.unsqueeze(0).expand(batch_size, -1, -1)

    ax = (sigma_x * dt) / 2.0
    az = (sigma_z * dt) / 2.0
    
    c1x = 2.0 / (1 + ax)
    c2x = (1 - ax) / (1 + ax)
    c3x = (v_batch**2 * dt**2) / (1 + ax)
    
    c1z = 2.0 / (1 + az)
    c2z = (1 - az) / (1 + az)
    c3z = (v_batch**2 * dt**2) / (1 + az)
    
    return c1x, c2x, c3x, c1z, c2z, c3z

def wave_sim_batched(v_in_batch: np.ndarray, src_z_batch: np.ndarray, src_x_batch: np.ndarray):

    batch_size = v_in_batch.shape[0]
    
    v_tensor = torch.from_numpy(v_in_batch).to(device)
    v_interp = nn.functional.interpolate(v_tensor, size=(target_width, target_width), mode="nearest")
    v_padded = nn.functional.pad(v_interp, (pml_width, pml_width, pml_width, pml_width), "replicate")
    v = v_padded.squeeze(1) # remove channel

    source = ricker_wavelet(f0, t_vec.cpu().numpy())
    
    src_z_tensor = torch.tensor(src_z_batch, dtype=torch.long, device=device)
    src_x_tensor = torch.tensor(src_x_batch, dtype=torch.long, device=device)


    c1x, c2x, c3x, c1z, c2z, c3z = create_pml_profiles_batched(batch_size, nz, nx, pml_width, pml_amp, dt, v)

    # batch_size, nz, nx
    p       = torch.zeros(batch_size, nz, nx, device=device)
    px      = torch.zeros(batch_size, nz, nx, device=device)
    pz      = torch.zeros(batch_size, nz, nx, device=device)
    p_prev  = torch.zeros(batch_size, nz, nx, device=device)
    px_prev = torch.zeros(batch_size, nz, nx, device=device)
    pz_prev = torch.zeros(batch_size, nz, nx, device=device)
    # 6-th order 
    c_6th = torch.tensor([1/90.0, -3/20.0, 3/2.0, -49/18.0, 3/2.0, -3/20.0, 1/90.0], dtype=torch.float32, device=device)
    kernel_x_6th = c_6th.view(1, 1, 1, 7) / (dx**2)
    kernel_z_6th = c_6th.view(1, 1, 7, 1) / (dz**2)
    
    snapshots_list = []
    for it in range(nt):
        # 增加channel维度以满足conv2d输入要求 (B, 1, nz, nx)
        p_batch_conv_in = p.unsqueeze(1)

        lap_x = torch.nn.functional.conv2d(p_batch_conv_in, kernel_x_6th, padding=(0, 3)).squeeze(1)
        lap_z = torch.nn.functional.conv2d(p_batch_conv_in, kernel_z_6th, padding=(3, 0)).squeeze(1)
        
        # 更新波场分量，所有计算都是在批处理维度上自动进行的
        px_new = c1x * px - c2x * px_prev + c3x * lap_x
        pz_new = c1z * pz - c2z * pz_prev + c3z * lap_z
        
        p_new = px_new + pz_new

        # 创建一个全零的掩码张量
        source_mask = torch.zeros(batch_size, nz, nx, device=device)
        source_mask[torch.arange(batch_size), src_z_tensor, src_x_tensor] = 1.0
        # 将子波值广播并添加到所有批次的对应位置
        p_new += source_mask * source[it]

        # 更新历史波场
        px_prev.copy_(px)
        px.copy_(px_new)
        pz_prev.copy_(pz)
        pz.copy_(pz_new)
        p.copy_(p_new)

        # 存储快照，去除PML边界
        if it % 10 == 0:  
            snapshot_no_pml = p[:, pml_width:target_width + pml_width, pml_width:pml_width + target_width]
            snapshots_list.append(snapshot_no_pml.detach().clone())
    
    # 7. 后处理
    # 将快照列表堆叠成一个张量
    snapshots_tensor = torch.stack(snapshots_list, dim=1) # (B, nt_snapshots, H, W)
    
    # 调整尺寸 (B, nt_snapshots, H, W) -> (B*nt_snapshots, 1, H, W)
    b_size, nt_snap, h, w = snapshots_tensor.shape
    snapshots_reshaped = snapshots_tensor.view(b_size * nt_snap, 1, h, w)
    
    # 插值
    resampled_snapshots = nn.functional.interpolate(snapshots_reshaped, size=(res_width, res_width), mode='bilinear', align_corners=False)
    
    # 恢复形状 (B, nt_snapshots, res_width, res_width)
    final_res = resampled_snapshots.view(b_size, nt_snap, res_width, res_width)
    
    return final_res.cpu().numpy()


def getFilesFromPath(rootDir, num):
    files = sorted(os.listdir(rootDir))
    train_files = [os.path.join(rootDir, filename) for filename in files[:(num // 500)]]
    return train_files


def processBatchData(v_batched_in : np.ndarray, model_id : list ,isTrain = True):
    batch_size = v_batched_in.shape[0]
    if len(model_id) != batch_size:
        raise ValueError
    v_batched_in_Cal = np.repeat(v_batched_in, repeats= 4, axis=0)
    source_points = np.array(target_Source_Point)
    source_points = np.tile(source_points, reps=(batch_size, 1))
    z_batched_in = source_points[:,1]
    x_batched_in = source_points[:,0]
    wave_res = wave_sim_batched(v_batched_in_Cal ,z_batched_in, x_batched_in)


    numT = wave_res.shape[1]
    wave_res = wave_res.reshape(batch_size, 4, numT , res_width, res_width)
    
    for batch_num in range(batch_size):
        vel_path = f"{v_OUTPUT_DIR}{"train" if isTrain else "test"}/data/velData{model_id[batch_num]}.npy"
        np.save(vel_path, v_batched_in[batch_num])

        for source_shot in range(4):
            wave_snapshot = wave_res[batch_num][source_shot]
            if not isTrain:
                sp_path = f"{v_OUTPUT_DIR}{"train" if isTrain else "test"}/sp/ResModel{model_id[batch_num]}_source{source_shot}.npy"
                np.save(sp_path, wave_snapshot)

            freq_path = f"{v_OUTPUT_DIR}{"train" if isTrain else "test"}/freq/fftResModel{model_id[batch_num]}_source{source_shot}.npy"


            # b) 保存频域波场数据
            fft_wave = np.fft.rfft(wave_snapshot, axis=0)
            low_freq_data = fft_wave[:23, :, :]
            np.save(freq_path, low_freq_data)
            
            # c) 检查并保存速度场数据

            
    


def main():
    # --- 1. 创建输出目录 ---
    os.makedirs(os.path.join(v_OUTPUT_DIR, "train/freq"), exist_ok=True)
    os.makedirs(os.path.join(v_OUTPUT_DIR, "train/data"), exist_ok=True)
    os.makedirs(os.path.join(v_OUTPUT_DIR, "test/sp"), exist_ok=True)
    os.makedirs(os.path.join(v_OUTPUT_DIR, "test/data"), exist_ok=True)
    os.makedirs(os.path.join(v_OUTPUT_DIR, "test/freq"), exist_ok=True) # 为测试集创建freq目录
    print(f"数据将保存至: '{v_OUTPUT_DIR}'")

    # --- 2. 收集并划分任务 ---
    train_files = []
    for file_path in v_ROOT_DIR:
        train_files += getFilesFromPath(file_path, numFromAClass)
    
    print("开始收集所有任务...")
    overAllTask = []
    model_idx_counter = 0
    # 为每个速度模型分配一个唯一的ID，以便在打乱后仍能识别
    for file in tqdm(train_files, desc="加载速度模型文件"):
        v_field_numpy = np.load(file)
        for i in range(v_field_numpy.shape[0]):
            vel_model = v_field_numpy[i]
            task_info = {
                    "model_id": model_idx_counter,
                    "vel_model": vel_model,
                    "prop": file.split('/')[-3]
            }
            overAllTask.append(task_info)
            model_idx_counter += 1

    CA_data = [task for task in overAllTask if task["prop"] == "CA"]
    FFB_data = [task for task in overAllTask if task["prop"] == "FFB"]
    styleA_data = [task for task in overAllTask if task["prop"] == "style-A"]

    random.shuffle(CA_data)
    random.shuffle(FFB_data)
    random.shuffle(styleA_data)

    test_tasks, train_tasks = [], []

    # 每个任务代表一次模拟（一个模型，一个震源）
    # test_sample_num 指的是速度模型的数量，每个模型有4个震源
    test_tasks += (CA_data[:test_sample_num ] + FFB_data[:test_sample_num ] + styleA_data[:test_sample_num ])
    train_tasks += (CA_data[test_sample_num :] + FFB_data[test_sample_num :] + styleA_data[test_sample_num :])
    
    random.shuffle(test_tasks)
    random.shuffle(train_tasks)

    # 准备train数据
    for i in tqdm(range(0 , len(train_tasks) , BATCH_SIZE)):

        batch_cal = train_tasks[i: i + BATCH_SIZE]
        v_batch = np.array([task['vel_model'] for task in batch_cal])
        id_batch = [task['model_id'] for task in batch_cal]

        processBatchData(v_batched_in=v_batch, model_id= id_batch, isTrain=True)

    for i in tqdm(range(0 , len(test_tasks), BATCH_SIZE)):
        batch_cal = test_tasks[i : i + BATCH_SIZE]
        v_batch = np.array([task['vel_model'] for task in batch_cal])
        id_batch = [task['model_id'] for task in batch_cal]    
        processBatchData(v_batched_in=v_batch, model_id= id_batch, isTrain=False)

if __name__ == "__main__":
    main()

