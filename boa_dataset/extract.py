# 将npz中提供的标签对应的imgnames提取到一个txt中

import numpy as np
import os
##设置全部数据，不输出省略号
import sys

np.set_printoptions(threshold=sys.maxsize)
i=0
for i in range(1,39):
  dex = "/home/neu307/liumeilin/datasets/3dpw/stressfile/"+ str(i) +r".npy"
  boxes = np.load(dex)

  np.savetxt('/home/neu307/liumeilin/datasets/3dpw/stressfile/boxes.txt', boxes, fmt='%s', newline='\n')
  print('---------------------boxes--------------------------')

with open("/home/neu307/liumeilin/datasets/3dpw/stressfile/boxes.txt", "r") as f:
  data = f.readlines()
  print(data)