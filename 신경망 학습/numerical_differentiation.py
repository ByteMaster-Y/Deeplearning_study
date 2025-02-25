import numpy as np
import matplotlib.pylab as plt

#이 코드들은 딥러닝에서 기울기(Gradient) 를 구하는 핵심 개념인 수치 미분(Numerical Differentiation) 을 설명하는 예제입니다.

# 📌 왜 중요한가?
# 딥러닝에서 최적화(Optimization) 는 주어진 손실 함수(Loss Function)를 최소화하는 과정입니다. 
# 이를 위해 기울기(Gradient) 를 계산하여 경사 하강법(Gradient Descent) 으로 가중치를 업데이트합니다.
# 하지만 실제로 미분을 정확한 해석적인 방식(Analytical Differentiation) 으로 구하기 어려운 경우가 많습니다. 
# 그래서 미분을 근사적으로 구하는 방법인 수치 미분(Numerical Differentiation) 이 유용할 수 있습니다.



# 미분
def numerical_diff(f, x):
    h = 10e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

# x = np.arange(0.0,20.0,0.1)
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x, y)
# plt.show()

# 편미분
# 인수들의 제곱 합을 계산하는 식

def function_2(x):
    return x[0]**2 + x[1]**2

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp1, 3.0))