# coding=utf-8

import numpy as np
import operator
from preprocess import Preproce as pp

class KNN(object):
	def __init__(self):
		pass
	
	def test(self, train_x, train_y, test_x, test_y, K):
		m = train_x.shape[0]
		# 欧式距离字典
		EuDistance = dict()
		# 余弦距离字典
		cosDistance = dict()
		m_test = test_x.shape[0]
		euClassList = [0,0,0]
		cosClassList = [0,0,0]
		euAccNum = 0
		cosAccNum = 0
		for ii in range(m_test):
			test = test_x[ii]
			# 欧式距离计算
			for i in range(m):
				 eudis = np.sqrt(np.sum(np.square(test-train_x[i])))
				 # print eudis
				 # print int(train_y[i])
				 EuDistance[i] = eudis
			EuDistanceList = sorted(EuDistance.items(), key=operator.itemgetter(1))
			for j in range(K):
				train_x_index = EuDistanceList[j][0]
				if (train_y[train_x_index] == 1):
					euClassList[0] += 1
				if (train_y[train_x_index] == 2):
					euClassList[1] += 1
				if (train_y[train_x_index] == 3):
					euClassList[2] += 1
			num = euClassList.index(max(euClassList)) + 1
			if (num == int(test_y[ii])):
				euAccNum += 1
			# 余弦距离计算
			for k in range(m):
				cosdis = np.dot(test, train_x[k])/(np.linalg.norm(test) * np.linalg.norm(train_x[k]))
				cosDistance[k] = cosdis

			cosDistanceList = sorted(cosDistance.items(), key=operator.itemgetter(1))
			for j in range(K):
				train_x_index = cosDistanceList[j][0]
				if (train_y[train_x_index] == 1):
					cosClassList[0] += 1
				if (train_y[train_x_index] == 2):
					cosClassList[1] += 1
				if (train_y[train_x_index] == 3):
					cosClassList[2] += 1
			cosNum = cosClassList.index(max(cosClassList)) + 1
			if (cosNum == int(test_y[ii])):
				cosAccNum += 1
		euAcc = float(euAccNum)/len(test_x)
		cosAcc = float(cosAccNum)/len(test_x)
		return euAcc, cosAcc

if __name__ == "__main__":
	knn = KNN()
	with open("./data.txt", "r") as f:
		data = f.readlines()
		for i in range(len(data)):
			data[i] = data[i].replace("\n","").split(",")
	with open("./target.txt", "r") as e:
		target = e.readlines()
		for i in range(len(target)):
			target[i] = target[i].replace("\n", "").split(",")
	
	train_x, test_x, train_y, test_y = pp().train_test_split(data, target, test_size=0.1)
	train_x = np.array(train_x, dtype=np.float32)
	train_y = np.array(train_y, dtype=np.int)
	test_x = np.array(test_x, dtype=np.float32)
	test_y = np.array(test_y, dtype=np.int)

	euAcc, cosAcc = knn.test(train_x, train_y, test_x, test_y, 20)
	print u"欧式距离结果", euAcc
	print u"余弦距离结果", cosAcc