# coding=utf-8

from preprocess import *
import numpy as np

def sigmoid(Z):
	value = 1. / (1 + np.exp(-Z))
	return value

def relu(Z):
	value = np.maximum(0, Z)
	return value

def leeky_relu(Z):
	value = np.maximum(0.01 * Z, Z)
	return value

# 初始化参数
def initialize_parameters(layers_dims):
	np.random.seed(1)
	parameters = {}
	l = len(layers_dims)
	for i in range(l-1):
		parameters["W" + str(i+1)] = np.random.randn(layers_dims[i+1], layers_dims[i])/ np.sqrt(layers_dims[i])
		parameters["b" + str(i+1)] = np.zeros((layers_dims[i+1], 1))
		assert (parameters["W" + str(i+1)].shape == (layers_dims[i+1], layers_dims[i]))
	return parameters

# 前向传播
# 输入参数：train_x, parameters
# 输出：caches（Z_l,A_prev_l-1, W_l, b_l的缓存）不包括X
def forward_propagation(X, parameters, activation):
	caches = []
	L = len(parameters) // 2
	A_prev = X
	for i in range(L):
		W = parameters["W" + str(i+1)]
		b = parameters["b" + str(i+1)]
		Z = W.dot(A_prev) + b
		cache = (Z, A_prev, W, b)
		if i != L-1:
			if activation == "relu":
				A_prev = relu(Z)
			if activation == "leeky_relu":
				A_prev = leeky_relu(Z)
		else:
			A_prev = sigmoid(Z)
		caches.append(cache)
	AL = A_prev
	return AL, caches

# 计算成本函数值
# 输入：A_L, Y
# 输出：cost
def compute_cost(A_L, Y):
	m = Y.shape[1]
	# cost = - np.sum(np.multiply(Y, np.log(A_L)) + np.multiply(1-Y, np.log(1-A_L))) / m
	cost = (1. / m) * (-np.dot(Y, np.log(A_L).T)-np.dot(1-Y, np.log(1-A_L).T))
	cost = np.squeeze(cost)
	assert (cost.shape == ())
	return cost

# dZ
# 输入：dA, Z, activation
# 输出：dZ
def linear_activation(dA, Z, activation):
	if activation == "sigmoid":
		s = sigmoid(Z)
		dZ = dA * s * (1-s)

	if activation == "relu":
		dZ = np.array(dA, copy=True)
		dZ[Z <= 0] = 0

	if activation == "leeky_relu":
		dZ = np.array(dA, copy=True)
		dZ[Z <= 0] *= 0.01
	assert (dZ.shape == dA.shape)
	return dZ

def linear_backward(dZ, A_prev, W, b):
	m = A_prev.shape[1]
	dW = np.dot(dZ, A_prev.T) / m
	db = np.sum(dZ, axis=1, keepdims=True) / m
	dA_prev = np.dot(W.T, dZ)
	# print(dZ.shape)
	# print(A_prev.shape)
	# print(dW.shape)
	# print(W.shape)
	# print(1)
	assert (db.shape == b.shape)
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	return dW, db, dA_prev

# 后向传播
# 输入：X, Y,caches, parameters
# 输出：grads(权重和偏离值下降的梯度)
def backward_propagation(AL, Y,caches, activation):
	grads = {}
	L = len(caches)
	Y = Y.reshape(AL.shape)
	Z, A_prev, W, b = caches[-1]
	dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
	dZ = linear_activation(dAL, Z, activation="sigmoid")
	grads["dW" + str(L)], grads["db" + str(L)], dA_prev = linear_backward(dZ, A_prev, W, b)

	for i in reversed(range(L-1)):
		Z, A_prev, W, b = caches[i]
		dZ = linear_activation(dA_prev, Z, activation)
		grads["dW" + str(i+1)], grads["db" + str(i+1)], dA_prev = linear_backward(dZ, A_prev, W, b)
	return grads

# 更新权重和偏离值
# 输入：parameters, grads, learning_rate
# 输出：parameters
def update_parameters(parameters, grads, learning_rate):
	L = len(parameters) // 2
	for i in range(L):
		parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - learning_rate * grads["dW" + str(i+1)]
		parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - learning_rate * grads["db" + str(i+1)]
	return parameters

# 预测
def predict(X, y, parameters, activation):
	m = X.shape[1]
	n = len(parameters) // 2
	p = np.zeros((1, m))
	probas, caches = forward_propagation(X, parameters, activation)
	for i in range(0, probas.shape[1]):
		if probas[0, i] > 0.5:
			p[0, i] = 1
		else:
			p[0, i] = 0
	print("Accuracy: " + str(np.sum((p == y) / m)))
	return p

# L层模型
def L_layers_model(train_x, train_y, num_iterations, learning_rate, activation="relu"):
	parameters = initialize_parameters(layers_dims)
	for i in range(num_iterations):
		AL, caches = forward_propagation(train_x, parameters, activation)
		cost = compute_cost(AL, train_y)
		grads = backward_propagation(AL, train_y, caches, activation)
		parameters = update_parameters(parameters, grads, learning_rate)
		if i % 100 == 0:
			print("cost in iteration %d is : %f" % (i, cost))
	return parameters


if __name__ == "__main__":
	# 导入数据并处理
	# 12288是train_x.shape[0], 1是train_y.shape[0]
	layers_dims = [12288, 20, 7, 5, 1]
	np.random.seed(1)
	train_x, train_y, test_x, test_y = load_dataset()
	train_x_flatten = train_x.reshape(train_x.shape[0], -1).T
	test_x_flatten = test_x.reshape(test_x.shape[0], -1).T

	train_x = train_x_flatten / 255
	test_x = test_x_flatten / 255
	activation = "relu"
	parameters = L_layers_model(train_x, train_y, num_iterations=2500,
								learning_rate=0.0075, activation=activation)
	predict(test_x, test_y, parameters, activation)

