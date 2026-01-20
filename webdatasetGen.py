import torch, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time, os, pdb
import torch.nn as nn
from tqdm import tqdm
import webdataset as wds  # --- 新增: 引入 WebDataset ---
# --- 1. 全局配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的计算设备: {device}")

# 物理和模拟参数
pml_width = 50                      # PML边界层宽度  
pml_amp = 1000.0                    # PML衰减参数
dz, dx = 12.0, 12.0                # z, x 上的网格大小, 单位m
target_width = 125                  # 实际模拟的时候的网格大小, 这里模拟125 x 125的网格, 实际上是1750mx1750m的模拟场
nt = 1000                           # 要模拟多少个时间步
dt = 0.001                          # 时间步长大小 单位s
f0 = 8.0                            # 生成Ricker子波的频率 hz
res_width = 125                      # 输出结果的尺寸, 输出的结果将会降采样到这个大小
test_sample_num = 500 * 4             # 划分数据集的时候, 测试集的速度场数量
numFromAClass = 2000 * 4              # 一个类别的总共的速度场的数量
BATCH_SIZE = 32                     # 批处理计算的大小
# 震源点定义
raw_target_point = [[25, 0]]
target_Source_Point =  [[point[0] + pml_width, pml_width] for point in raw_target_point]
# target_Source_Point = [[int(point[0] * 25 / dz + 0.5) + pml_width, pml_width] for point in raw_target_point]
SHARD_MAX_SIZE = 1e10
MAX_COUNT = 500
# 时间向量和计算域尺寸
t_vec = torch.arange(0, nt * dt, dt, device=device)
nz, nx = target_width + pml_width * 2, target_width + pml_width * 2

# 数据路径
v_ROOT_DIR = ["/home/zhangzhe/data/velField/CA/model", "/home/zhangzhe/data/velField/style-A/model", "/home/zhangzhe/data/velField/FFB/model"]
# v_ROOT_DIR = ["/home/zhangzhe/data/velField/FFB/model"]
v_OUTPUT_DIR = "/home/zhangzhe/data/newGenData_singal/"



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
    
    # 1. 准备速度模型
    v_tensor = torch.from_numpy(v_in_batch).to(device)
    # 确保维度为 (B, 1, H, W) 以进行插值
    if v_tensor.dim() == 3:
        v_tensor = v_tensor.unsqueeze(1)
        
    v_interp = nn.functional.interpolate(v_tensor, size=(target_width, target_width), mode="nearest")
    v_padded = nn.functional.pad(v_interp, (pml_width, pml_width, pml_width, pml_width), "replicate")
    v = v_padded.squeeze(1) # (B, nz, nx)

    # 2. 准备震源和坐标
    source = ricker_wavelet(f0, t_vec.cpu().numpy())
    
    src_z_tensor = torch.tensor(src_z_batch, dtype=torch.long, device=device)
    src_x_tensor = torch.tensor(src_x_batch, dtype=torch.long, device=device)

    # [安全修复] 防止坐标越界导致 CUDA error

    # 用于索引 batch 的辅助张量
    batch_indices = torch.arange(batch_size, device=device)

    # 3. 计算 PML 参数
    c1x, c2x, c3x, c1z, c2z, c3z = create_pml_profiles_batched(batch_size, nz, nx, pml_width, pml_amp, dt, v)

    # 4. 初始化波场
    p       = torch.zeros(batch_size, nz, nx, device=device)
    px      = torch.zeros(batch_size, nz, nx, device=device)
    pz      = torch.zeros(batch_size, nz, nx, device=device)
    p_prev  = torch.zeros(batch_size, nz, nx, device=device)
    px_prev = torch.zeros(batch_size, nz, nx, device=device)
    pz_prev = torch.zeros(batch_size, nz, nx, device=device)
    
    # --- 修改部分：4阶有限差分系数 ---
    # Coefficients: [-1/12, 4/3, -5/2, 4/3, -1/12]
    # Center is -5/2, neighbors are 4/3 and -1/12
    c_4th = torch.tensor([-1/12.0, 4.0/3.0, -5.0/2.0, 4.0/3.0, -1/12.0], dtype=torch.float32, device=device)
    
    # Kernel size = 5
    kernel_x_4th = c_4th.view(1, 1, 1, 5) / (dx**2)
    kernel_z_4th = c_4th.view(1, 1, 5, 1) / (dz**2)
    
    snapshots_list = []
    
    for it in range(nt):
        # 增加channel维度 (B, 1, nz, nx)
        p_batch_conv_in = p.unsqueeze(1)

        lap_x = torch.nn.functional.conv2d(p_batch_conv_in, kernel_x_4th, padding=(0, 2)).squeeze(1)
        lap_z = torch.nn.functional.conv2d(p_batch_conv_in, kernel_z_4th, padding=(2, 0)).squeeze(1)
        
        # 更新波场分量
        px_new = c1x * px - c2x * px_prev + c3x * lap_x
        pz_new = c1z * pz - c2z * pz_prev + c3z * lap_z
        
        p_new = px_new + pz_new

        # --- 优化部分：震源注入 ---
        # 直接索引相加，代替创建全零 Mask (速度更快，显存更省)
        p_new[batch_indices, src_z_tensor, src_x_tensor] += source[it]

        # 更新历史波场
        px_prev.copy_(px)
        px.copy_(px_new)
        pz_prev.copy_(pz)
        pz.copy_(pz_new)
        p.copy_(p_new)

        # 存储快照
        if it % 10 == 0:  
            snapshot_no_pml = p[:, pml_width:target_width + pml_width, pml_width:pml_width + target_width]
            snapshots_list.append(snapshot_no_pml.detach().clone())
    
    # 5. 后处理
    snapshots_tensor = torch.stack(snapshots_list, dim=1) # (B, nt/10, H, W)
    b_size, nt_snap, h, w = snapshots_tensor.shape
    snapshots_reshaped = snapshots_tensor.view(b_size * nt_snap, 1, h, w)
    if res_width != target_width:
        resampled_snapshots = nn.functional.interpolate(snapshots_reshaped, size=(res_width, res_width), mode='bilinear', align_corners=False)
    else :
        resampled_snapshots = snapshots_reshaped
    final_res = resampled_snapshots.view(b_size, nt_snap, res_width, res_width)
    
    return final_res.cpu().numpy()

def getFilesFromPath(rootDir, num):
    files = sorted(os.listdir(rootDir))
    train_files = [os.path.join(rootDir, filename) for filename in files[:(num // 500)]]
    return train_files

# --- 修改: 增加 writer 参数，不再进行 np.save，而是写入 webdataset ---
def processBatchData(v_batched_in: np.ndarray, model_id: list, writer: wds.ShardWriter, isTrain=True):
    batch_size = v_batched_in.shape[0]
    if len(model_id) != batch_size:
        raise ValueError
    
    # chaneg len(raw_target_point) source point
    v_batched_in_Cal = np.repeat(v_batched_in, repeats=len(raw_target_point), axis=0)
    source_points = np.array(target_Source_Point)
    source_points = np.tile(source_points, reps=(batch_size, 1))
    z_batched_in = source_points[:, 1]
    x_batched_in = source_points[:, 0]
    
    # 执行模拟
    with torch.no_grad():
        wave_res = wave_sim_batched(v_batched_in_Cal, z_batched_in, x_batched_in)
    # 结果 Reshape: (Batch, Sources, Time, H, W)
    numT = wave_res.shape[1]
    wave_res = wave_res.reshape(batch_size, len(raw_target_point) , numT, res_width, res_width)
    
    # 写入 WebDataset
    for batch_num in range(batch_size):
        # 1. 基础样本信息
        # __key__ 必须是字符串，通常是唯一ID
        sample = {
            "__key__": str(model_id[batch_num]),
            "vel.npy": v_batched_in[batch_num], # 自动序列化 numpy array
        }
        
        # 2. 遍历4个震源的数据
        for source_shot in range(len(raw_target_point)):
            wave_snapshot = wave_res[batch_num][source_shot]
            
            # 频域处理
            fft_wave = np.fft.rfft(wave_snapshot, axis=0)
            low_freq_data = fft_wave[:30, :, :]
            
            # 将频域数据添加到样本字典中
            # 命名习惯: source{i}.freq.npy
            sample[f"source{source_shot}.freq.npy"] = low_freq_data    

        # 4. 写入当前样本到 Tar 包
        writer.write(sample)


def main():
    # --- 1. 创建输出目录 (WebDataset Shards 目录) ---
    train_dir = os.path.join(v_OUTPUT_DIR, "train_shards")
    test_dir = os.path.join(v_OUTPUT_DIR, "test_shards")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"WebDataset Shards 将保存至: '{train_dir}' 和 '{test_dir}'")

    # --- 2. 初始化 WebDataset Writer ---
    # pattern: train/train-000000.tar
    train_pattern = os.path.join(train_dir, "train-%06d.tar")
    test_pattern = os.path.join(test_dir, "test-%06d.tar")

    train_writer = wds.ShardWriter(train_pattern, maxcount = MAX_COUNT, maxsize = SHARD_MAX_SIZE)
    test_writer = wds.ShardWriter(test_pattern, maxcount= MAX_COUNT, maxsize = SHARD_MAX_SIZE)

    # --- 3. 收集并划分任务 ---
    train_files = []
    for file_path in v_ROOT_DIR:
        train_files += getFilesFromPath(file_path, numFromAClass)
    
    print("开始收集所有任务...")
    overAllTask = []
    model_idx_counter = 0
    # 为每个速度模型分配一个唯一的ID
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

    test_tasks += (CA_data[:test_sample_num ] + FFB_data[:test_sample_num ] + styleA_data[:test_sample_num ])
    train_tasks += (CA_data[test_sample_num :] + FFB_data[test_sample_num :] + styleA_data[test_sample_num :])
    
    random.shuffle(test_tasks)
    random.shuffle(train_tasks)

    # --- 4. 执行处理 ---
    
    print("正在处理 Train 数据集...")
    for i in tqdm(range(0 , len(train_tasks) , BATCH_SIZE)):
        batch_cal = train_tasks[i: i + BATCH_SIZE]
        v_batch = np.array([task['vel_model'] for task in batch_cal])
        id_batch = [task['model_id'] for task in batch_cal]

        # 传入 train_writer
        processBatchData(v_batched_in=v_batch, model_id=id_batch, writer=train_writer, isTrain=True)

    print("正在处理 Test 数据集...")
    for i in tqdm(range(0 , len(test_tasks), BATCH_SIZE)):
        batch_cal = test_tasks[i : i + BATCH_SIZE]
        v_batch = np.array([task['vel_model'] for task in batch_cal])
        id_batch = [task['model_id'] for task in batch_cal]    
        
        # 传入 test_writer
        processBatchData(v_batched_in=v_batch, model_id=id_batch, writer=test_writer, isTrain=False)

    # --- 5. 关闭 Writer ---
    train_writer.close()
    test_writer.close()
    print("数据生成完成。")

if __name__ == "__main__":
    main()
