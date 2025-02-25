import numpy as np

# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7  # 수정: theta에 올바른 값 대입
#     tmp = x1 * w1 + x2 * w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    w = np.array([0.5, 0.5])
    b = -0.7  # 편향
    return 1 if np.sum(w * np.array([x1, x2])) + b <= 0 else 0

def OR(x1, x2):
    w = np.array([0.5, 0.5])
    b = -0.2  # 편향
    return 1 if np.sum(w * np.array([x1, x2])) + b > 0 else 0

def XOR(x1, x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1, s2)
    return y

# 선형 분리가 불가능한 문제:
# AND, OR, NAND 같은 기본 논리 함수들은 선형적으로 분리 가능한 문제들이에요. 즉, 입력값들을 선형 경계로 나눌 수 있습니다.
# 하지만 XOR은 선형 분리가 불가능한 문제예요. 예를 들어, XOR(0,0) = 0, XOR(0,1) = 1, XOR(1,0) = 1, XOR(1,1) = 0인데, 이 값들을 2D 평면에 그래프를 그려보면 선으로 구분할 수 없다는 점에서 선형 분리가 불가능하다고 말할 수 있습니다.
# 다층 퍼셉트론(Multi-layer Perceptron, MLP):
# XOR 문제를 해결하려면 다층 신경망이 필요합니다. 단일층 퍼셉트론은 XOR을 해결할 수 없지만, 다층 퍼셉트론을 사용하면 XOR 문제를 해결할 수 있어요. 즉, XOR 문제는 비선형 활성화 함수를 가진 다층 신경망에서 잘 해결됩니다.
# 기존의 게이트들을 이용한 조합:
# XOR은 **기존의 논리 게이트(AND, OR, NAND)**를 조합해서 만들어낼 수 있는 함수이지만, 여전히 이 조합 자체는 선형적으로 분리할 수 없습니다. 즉, XOR은 단순히 기존의 논리 연산들을 조합하는 것만으로는 해결할 수 없고, 비선형 모델이나 다층 신경망을 통해서 해결해야 하는 문제입니다.  

print(AND(0, 0))  # 출력값: 0
print(AND(1, 0))  # 출력값: 0
print(AND(0, 1))  # 출력값: 0
print(AND(1, 1))  # 출력값: 1


print(XOR(0,0))
print(XOR(1,0))
