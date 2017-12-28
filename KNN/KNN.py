# coding=utf-8

import numpy as np
from preprocess import Preproce as pp

class KNN(object):
	















if __name__ == "__main__":
	with open("./data.txt", "r") as f:
		data = f.readlines()
		for i in range(len(data)):
			data[i] = data[i].replace("\n","")
	with open("./target.txt", "r") as e:
		target = e.readlines()
		for i in range(len(target)):
			target[i] = target[i].replace("\n", "")
	train_x, test_x, train_y, test_y = pp().train_test_split(data, target, test_size=0.4)