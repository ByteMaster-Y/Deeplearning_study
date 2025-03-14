# Mean Squared Error (MSE) 평규 제곱 오차
# 오버피팅: 한 데이터셋에만 지나치게 최적화된 상태를 오버피팅이라함
# 손실 함수: 신경망 성능의 '나쁨'을 나타내는 지표로, 현재 신경망이 훈련 데이터를 얼마나 잘 처리하지 못하느냐를 나타냄. '성능 나쁨'을 지표로 한다니
# 무언가 부자연스럽다고 생각할지 모르지만, 손실 함수에 마아너스만 곱하면 얼마나 나쁘지 않냐로 변신함. 즉 신경망 학습에 사용하는 지표를 손실 함수라함
import numpy as np 

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) # delta는 아주 작은 값을 더해서 절대 0이 되지 않도록 하는거임

t = [0, 0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
# 예1 : '2'일 확률이 가장 높다고 추정함 (0.6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0 ,0.1 ,0.0 ,0.0]
# print(mean_squared_error(np.array(y), np.array(t)))
# 예2 : '7'일 확률이 가장 높다고 추정함(0.6)
# y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# print(mean_squared_error(np.array(y), np.array(t)))

# 교차 엔트로피 오차 (Cross entropy error, CEE)
# 교차 엔트로피는 모델이 예측한 확률과 실제 정답 사이의 차이를 측정하는 손실 함수로, 분류 문제에서 자주 사용됩니다.
print(cross_entropy_error(np.array(y), np.array(t)))
# 위의 사실로 알 수 있는 점: 첫번째 예는 정답일 때의 출력이 0.6인 경우로, 이떄의 교차 엔트로피 오차는 약0.51 입니다
# 그다음 정답일 때의 출력이 0.1 더낮은 경우로 이떄의 교차 엔트로피 오차는 무려 2.3입니다. 즉 오차값이 더작은 첫번째 추정이 정답일 가능성이 높습니다.

# mnist가 60,000개가 넘는데 전부 돌려서 평균 손실 함수를 구하면 시간이 오래걸리므로
# 그중에 100장을 무작위로 뽑아서 100장만을 사용하여 학습시키는 것 -> 미니배치라함
