# 입력이 0을 넘으면 그 입력을 그대로 출력하고, 0이하이면 0을 출력한다!
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_function(x): # 항등함수
    return x

# 다차원 배열
# A = np.array([1, 2, 3, 4])
# print(np.ndim(A)) # 차원의 수!
# print(A.shape) # 배열의 형상 -> 원소 4개로 이루어져있다.
# B = np.array([[1,2],[3,4],[5,6]])
# print(B)
# print(np.ndim(B))
# print(B.shape) # 항상 shape은 튜플로 반환한다! 항상 동일한 형태로 결과를 반환하기 위해서

# 행렬의 내적
# A = np.array([[1,2],[3,4]])
# print(A.shape)
# B = np.array([[5,6], [7,8]])
# print(B.shape)
# print(np.dot(A,B)) # 행렬의 내적을 np.dot으로 구한다.

# 행렬의 곱에서는 대응하는 차원의 원소 수를 일치시켜라!
# A가 2차원 B가 1차원 배열일 때도 대응하는 차원의 원소 수를 일치시켜라
A = np.array([[1,2], [3,4], [5,6]])
print(A.shape)
B = np.array([7,8])
print(np.dot(A,B))

# 신경망의 내적
# X = np.array([1,2])
# print(X.shape)
# W = np.array([[1,3,5],[2,4,6]])
# print(W)
# print(W.shape)
# Y = np.dot(X,W)
# print(Y)

# 3층 신경망 구현하기
# 입력층은 2개, 은닉층은 3개 두번째 은닉층은 2개 출력층은 2개의 뉴런으로 구성
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + B1
# print(A1)
Z1 = sigmoid(A1)
# print(Z1)

# 1층에서 2층으로의 신호 전달
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])
A2 = np.dot(Z1,W2) + B2
Z2 = sigmoid(A2)
print(A2)

# 항등함수는 입력을 그대로 출력하는 함수
W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print(Y)

# 출력층의 활성화 함수는 풀고자하는 문제의 성질에 맞게 정한다. 예를들어 회귀에는 항등함수를, 2클래스 분류에는 시그모이드, 다중 클래스 분류에는 소프트맥스를 쓰는게 일반적
