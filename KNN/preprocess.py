# coding=utf-8

import random

class Preproce(object):
	def getData(self):
		with open("./wine.txt", "r") as f:
			allData = f.readlines()
		for j in range(len(allData)):
			dataList = allData[j].replace("\n", "").split(",")
			with open("./target.txt", "a") as t:
				t.writelines(dataList[0])
				if (j != (len(allData)-1)):
					t.writelines("\n")
			with open("./data.txt", "a") as d:
				for i in range(1,len(dataList)):
					d.writelines(dataList[i])
					if (i != (len(dataList)-1)):
						d.writelines(",")
				if (j != (len(allData)-1)):
					d.writelines("\n")


	def train_test_split(self, data, target, test_size=0.0):
		length = len(data)
		train_size_len = int(len(data) * (1 - test_size))
		randList = random.sample(range(0,len(data)), train_size_len)
		train_x = [data[i] for i in randList]
		train_y = [target[i] for i in randList]
		test_x = [data[j] for j in range(length) if j not in randList]
		test_y = [target[j] for j in range(length) if j not in randList]
		return train_x, test_x, train_y, test_y






if __name__ == "__main__":
	# pp = Preproce()
	# with open("./data.txt", "r") as f:
	# 	data = f.readlines()
	# with open("./target.txt", "r") as e:
	# 	target = e.readlines()
	# train_x, test_x, train_y, test_y=pp.train_test_split(data, target, 0.4)

	# for i in range(len(train_x)):
	# 	print train_x[i], "-",train_y[i]
	# from collections import Counter
	# a = ['sda','sda','asd','dg']
	# b = Counter(a)
	# print max(b.values())
	a = {
		"a":1,
		"b":2
	}
	print a.index(a)