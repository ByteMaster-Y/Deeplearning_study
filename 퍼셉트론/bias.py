import numpy as np
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7 # 편향
print(w*x)
print(np.sum(w*x) + b)

# 이 코드로 알 수 있는 것:
# 입력값, 가중치, 편향을 이용한 선형 변환을 확인할 수 있습니다.
# w * x와 np.sum(w * x)는 퍼셉트론에서 각 입력값에 가중치를 곱한 뒤 그 합을 구하는 과정입니다.
# 편향을 더하는 과정은 퍼셉트론이 출력값을 결정할 때 중요한 요소입니다. 이 값이 활성화 함수(예: 계단 함수, 시그모이드 등)에 입력되면 최종 출력이 결정됩니다.
