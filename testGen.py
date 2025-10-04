from utils.video import create_video_from_numpy
import numpy as np
import pdb
from utils.drawVel import plot_array_with_highlight
vel_data = np.load("/home/niuma009/data/newGenData/test/data/velData10586.npy")
vel_data = vel_data.squeeze(0)
plot_array_with_highlight(vel_data, (0, 51), "generated.png")
genData = np.load("/home/niuma009/data/newGenData/test/sp/ResModel10586_source3.npy")
create_video_from_numpy(genData, "generated.mp4", 10)