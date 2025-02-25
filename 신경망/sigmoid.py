import numpy as np
import matplotlib.pylab as plt

# def step_function(x):
#     y = x > 0
#     return y.astype(np.int)

# print(step_function(np.array([1, -1, 0, 2, -3])))

def step_function(x): # 시그모이드 활성화 함수
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1) # -5 -> 5.0까지 0.1을 간격으로 넘파이 생성
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

##### 2월22일 2번째 딥러닝 공부
# 계단 함수와 시그모이드 함수는 둘다 비선형 함수이다.
# 신경망에서는 활성화 함수로 반드시 비선형을 써야 한다.
# -> 층을 쌓는 혜택을 얻고 싶다면 활성화 함수로는 반드시 비선형 함수로 사용해야 한다.
