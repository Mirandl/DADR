# -*- coding: UTF-8 -*-
# !/usr/bin/env python
import shutil

# 根据txt中文件的名字批量提取对应的文件名并保存到另一个文件夹

data = []
for line in open("/home/neu307/liumeilin/datasets/3dpw/stressfile/boxes.txt","r"):
    # 设置文件对象并读取每一行文件
    data.append(line)

for a in data:
    src = '/home/neu307/liumeilin/datasets/3dpw/{}'.format(a[:-1])
    dst = '/home/neu307/liumeilin/datasets/3dpw_d/{}'.format(a[:-1])
    # shutil.copyfile(src, dst)
    shutil.copy(src, dst)
