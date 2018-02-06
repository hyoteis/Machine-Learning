# coding: utf-8

import h5py
import numpy as np

# 导入数据
def load_dataset():
    # 训练数据集读取
    train_file = h5py.File("./datasets/train_catvnoncat.h5", "r")
    train_x_orig = train_file["train_set_x"][:]
    train_y_orig = train_file["train_set_y"][:]
    
    # 测试数据集读取
    test_file = h5py.File("./datasets/test_catvnoncat.h5", "r")
    test_x_orig = test_file["test_set_x"][:]
    test_y_orig = test_file["test_set_y"][:]
    classes = test_file["list_classes"][:]
    
    train_y_orig = train_y_orig.reshape((1, train_y_orig.shape[0]))
    test_y_orig = test_y_orig.reshape((1, test_y_orig.shape[0]))
    
    return train_x_orig, train_y_orig, test_x_orig, test_y_orig
