import numpy as np
import matplotlib.pylab as plt

# 이 코드는 경사 하강법(Gradient Descent) 을 구현한 코드입니다.
# 경사 하강법은 기울기(Gradient)를 이용하여 함수의 최솟값을 찾는 최적화 방법입니다.
# 딥러닝에서 손실 함수(Loss Function)를 최소화할 때 사용됩니다.

# 💡 경사 하강법을 감으로 이해하자!

# 기울기(Gradient)는 현재 위치에서 어디로 이동해야 할지 방향을 알려주는 나침반
# 학습률(Learning Rate)은 얼마나 빠르게 이동할지 조절하는 속도 조절기
# 경사 하강법은 기울기를 따라 점점 내려가면서 최솟값을 찾는 과정
# 📌 실제로 딥러닝에서는 자동 미분(Autograd)이 계산을 해줌
# → 우리가 직접 미분을 계산할 필요 없음!
# → PyTorch, TensorFlow 같은 라이브러리에서 다 처리해 줌!

# 🔹 결론: 수식 자체를 외울 필요는 없지만,
# 🔹 "왜 필요한지" 와 "어떻게 동작하는지" 는 이해해야 함! 😊

# 앞 절에서 x0, x1에 대한 편미분을 변수별로 따로 계산했음.
# x0, x1의 편미분을 동시에 계산하고 싶다면?
# x0 = 3, x1 = 4일 때 (x0, x1) 양쪽의 편미분을 묶어 벡터로 정리한 것을 기울기gradient라고 한다.
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원

    return grad


# f(x0, x1) = x0² + x1²
def function_2(x):
    return x[0]**2 + x[1]**2
    # or return np.sum(x**2)


print(numerical_gradient(function_2, np.array([3.0, 4.0])))  # [ 6.  8.]
print(numerical_gradient(function_2, np.array([0.0, 2.0])))  # [ 0.  4.]
print(numerical_gradient(function_2, np.array([3.0, 0.0])))  # [ 6.  0.]

# 4.4.1 경사법(경사 하강법)
# x0 = x0 - η*∂f/∂x0
# x1 = x1 - η*∂f/∂x1
# η(eta) : 갱신하는 양, 학습률learning rate
# 위 식을 반복


# f:최적화하려는 함수
# init_x : 초깃값
# lr : 학습률
# step_num : 반복횟수
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


# 경사법으로 f(x0, x1) = x0² + x1²의 최솟값을 구해라
init_x = np.array([-3.0, 4.0])
x, x_history = gradient_descent(function_2, init_x, lr=0.1)
print(x)  # [ -6.11110793e-10   8.14814391e-10]

# 학습률이 너무 큼
init_x = np.array([-3.0, 4.0])
x, x_history = gradient_descent(function_2, init_x, lr=10.0)
print(x)  # [ -2.58983747e+13  -1.29524862e+12] 발산함

# 학습률이 너무 작음
init_x = np.array([-3.0, 4.0])
x, x_history = gradient_descent(function_2, init_x, lr=1e-10)
print(x)  # [-2.99999994  3.99999992] 거의 변화 없음

# 그래프
init_x = np.array([-3.0, 4.0])
x, x_history = gradient_descent(function_2, init_x, lr=0.1, step_num=20)

plt.plot([-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5, 5], '--b')
plt.plot(x_history[:, 0], x_history[:, 1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()