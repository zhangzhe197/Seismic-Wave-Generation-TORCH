import numpy as np
import matplotlib.pyplot as plt

def plot_array_with_highlight(
    array: np.ndarray,
    point_coords: tuple,
    output_path: str,
    title: str = 'Array Visualization',
    cmap: str = 'viridis',
    marker_color: str = 'red',
    marker_size: int = 100
):
    """
    绘制一个二维NumPy数组的图像，并高亮显示其中的一个特定点，然后将图像保存到文件。

    参数:
    array (np.ndarray): 需要可视化的二维NumPy数组。
    point_coords (tuple): 一个包含(row, col)坐标的元组，指定要高亮显示的点。
    output_path (str): 保存图像的文件路径 (例如 'output.png', 'result.jpg')。
    title (str, optional): 图像的标题。默认为 'Array Visualization'。
    cmap (str, optional): Matplotlib的颜色映射方案。默认为 'viridis'。
    marker_color (str, optional): 高亮标记的颜色。默认为 'red'。
    marker_size (int, optional): 高亮标记的大小。默认为 100。
    """
    # 1. 输入验证
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("输入 'array' 必须是一个二维NumPy数组。")

    rows, cols = array.shape
    point_row, point_col = point_coords

    if not (0 <= point_row < rows and 0 <= point_col < cols):
        raise ValueError(f"坐标 {point_coords} 超出数组范围 (0, {rows-1}) 和 (0, {cols-1})。")

    # 2. 创建图像和坐标轴
    fig, ax = plt.subplots(figsize=(8, 6))

    # 3. 使用 imshow 绘制数组图像
    # im 是 imshow 返回的对象，用于后续创建 colorbar
    im = ax.imshow(array, cmap=cmap, interpolation='nearest')

    # 4. 在指定点上绘制一个醒目的标记
    # 注意：imshow 的坐标系中，x 对应列(col)，y 对应行(row)
    ax.scatter(
        x=point_col,
        y=point_row,
        s=marker_size,          # 标记大小
        c=marker_color,         # 标记颜色
        marker='o',             # 标记形状（圆形）
        edgecolors='white',     # 添加白色描边让标记更突出
        linewidths=1.5,
        label=f'Point ({point_row}, {point_col})' # 为图例添加标签
    )

    # 5. 添加图表元素，增强可读性
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("列 (Column Index)", fontsize=12)
    ax.set_ylabel("行 (Row Index)", fontsize=12)
    ax.legend() # 显示图例

    # 添加颜色条 (colorbar)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Array Values', fontsize=12)

    # 6. 保存图像
    try:
        # bbox_inches='tight' 会裁剪掉图像周围多余的白边
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图像已成功保存至: {output_path}")
    except Exception as e:
        print(f"保存图像失败: {e}")
    finally:
        # 7. 关闭图像，释放内存
        plt.close(fig)