import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np

# MNIST 데이터 불러오기 (손글씨 숫자 데이터셋)
(x_train, t_train), (x_test, t_test) = keras.datasets.mnist.load_data()

# 데이터 전처리
# 이미지 데이터를 (28x28)에서 1차원 (784)로 변환하고, [0, 1] 범위로 정규화
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

# 정답 레이블을 원-핫 인코딩 (10개의 클래스로 변환), 예를 들어, 숫자 3은 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]과 같이 변환됩니다.
t_train = to_categorical(t_train, 10)

def get_mini_batch(x, t, batch_size=10):
    """
    미니배치를 추출하는 함수
    :param x: 입력 데이터 (이미지 데이터)
    :param t: 정답 레이블 (원-핫 인코딩)
    :param batch_size: 미니배치 크기 (기본값: 10)
    :return: 미니배치 데이터 (입력 x_batch, 정답 t_batch)
    """
    train_size = x.shape[0]  # 전체 데이터 개수 (60,000), x_train.shape의 출력 결과는 (60000, 784)이야.
    batch_mask = np.random.choice(train_size, batch_size)  # 랜덤하게 batch_size개 선택
    x_batch = x[batch_mask]  # 선택된 데이터 추출 , x_batch는 실제 모델 학습에서 사용되는 입력 데이터입니다
    t_batch = t[batch_mask]  # 선택된 정답 레이블 추출
    return x_batch, t_batch

# 교차 엔트로피 손실 함수 (분류 문제에서 사용)
# 분류 문제에서 자주 사용되는 손실 함수입니다. 다중 클래스 분류에서 예측값과 실제값 간의 차이를 계산하는 데 사용됩니다.
loss_fn = keras.losses.CategoricalCrossentropy()

# 미니배치 추출 및 손실 계산
x_batch, t_batch = get_mini_batch(x_train, t_train, batch_size=10)  # 10개 샘플 선택
y_pred = np.random.rand(10, 10)  # 가상의 예측값 (랜덤 값, 실제 모델이 필요)
loss = loss_fn(t_batch, y_pred)  # 손실 함수 계산
# loss_fn(t_batch, y_pred)는 실제 레이블(t_batch)과 예측값(y_pred)을 비교하여 손실을 계산합니다.
print("교차 엔트로피 손실:", loss.numpy())  # 손실 값 출력

# 신경망을 학습할 때 정확도를 지표로 삼아서는 안된다. 정확도를 지표로 하면 매개변수의 미분이 대부분의 장소에서 0이 되기때문이다.
