# 텐서플로우 라이브러리와 Keras에서 모델과 레이어를 임포트합니다.
import tensorflow as tf
from tensorflow.keras import layers, models

# MNIST 데이터셋 로드
(x_train, t_train), (x_test, t_test) = tf.keras.datasets.mnist.load_data()
# x_train, t_train: 학습용 이미지와 레이블
# x_test, t_test: 테스트용 이미지와 레이블
# MNIST 데이터셋은 28x28 크기의 흑백 이미지와 각 이미지에 해당하는 숫자 레이블을 포함하고 있습니다.

# 데이터를 정규화 (0~255 사이의 값을 0~1 사이로 변환)
x_train = x_train / 255.0  # 학습용 이미지 데이터 정규화
x_test = x_test / 255.0    # 테스트용 이미지 데이터 정규화
# 정규화는 각 이미지의 픽셀 값을 255로 나누어 0과 1 사이의 값으로 변환하여, 모델이 학습을 더 잘 할 수 있게 합니다.

# 모델 정의 시작
model = models.Sequential()  # Sequential 모델은 레이어가 순차적으로 쌓이는 모델
# 이 모델은 층을 하나씩 쌓아가는 방식으로, 각 층은 이전 층의 출력을 입력으로 받습니다.

# 첫 번째 레이어: Flatten 레이어
model.add(layers.Flatten(input_shape=(28, 28)))  # 28x28 크기의 이미지를 1차원 벡터로 펼칩니다.
# 입력 데이터가 28x28 크기의 2D 이미지이므로, 이 레이어는 1D 벡터로 변환하여 모델에 전달됩니다.
# 결과적으로 28x28 크기의 이미지는 784개의 값(28*28)을 가지는 1차원 배열로 변환됩니다.

# 두 번째 레이어: Dense 은닉층
model.add(layers.Dense(128, activation='relu'))  # 128개의 뉴런을 가지는 은닉층을 추가, ReLU 활성화 함수 사용
# Dense 층은 완전 연결된 신경망 층으로, 각 입력값은 모든 뉴런과 연결됩니다.
# `128`은 이 레이어의 뉴런 수를 의미하며, `activation='relu'`는 활성화 함수로 ReLU(Rectified Linear Unit)를 사용하겠다는 뜻입니다.
# ReLU는 음수 값을 0으로 변환하고 양수는 그대로 출력하는 비선형 함수로, 주로 은닉층에서 사용됩니다.

# 세 번째 레이어: 출력층
model.add(layers.Dense(10, activation='softmax'))  # 10개의 뉴런을 가지는 출력층을 추가, Softmax 활성화 함수 사용
# 출력층은 10개의 뉴런을 가지며, 각 뉴런은 0~9까지의 클래스에 해당하는 확률을 출력합니다.
# `activation='softmax'`는 소프트맥스 함수로, 각 뉴런의 출력을 확률로 변환하여, 가장 큰 값을 갖는 클래스가 예측값이 됩니다.
# MNIST는 0부터 9까지의 숫자 이미지를 분류하므로, 출력층은 10개의 클래스를 가집니다.

# 모델 컴파일
model.compile(optimizer='adam',  # Adam 옵티마이저를 사용
              loss='sparse_categorical_crossentropy',  # 손실 함수로 sparse categorical crossentropy 사용
              metrics=['accuracy'])  # 모델 평가 지표로 정확도(accuracy) 사용
# `optimizer='adam'`: Adam(Adaptive Moment Estimation) 옵티마이저를 사용하여 가중치를 업데이트합니다.
# Adam은 현재 가장 많이 사용되는 최적화 알고리즘 중 하나입니다.
# `loss='sparse_categorical_crossentropy'`: 다중 클래스 분류 문제에서 사용되는 손실 함수입니다.
# `sparse_categorical_crossentropy`는 레이블이 원-핫 인코딩되지 않고 정수로 주어졌을 때 사용합니다.
# `metrics=['accuracy']`: 모델의 성능을 평가할 때 정확도를 기준으로 합니다.

# 모델 학습
model.fit(x_train, t_train, epochs=5)  # 학습 데이터를 이용해 모델을 5번 학습
# `model.fit()` 함수는 모델을 학습시키는 함수입니다.
# `x_train`은 학습용 이미지 데이터, `t_train`은 학습용 레이블 데이터를 사용합니다.
# `epochs=5`는 모델을 5번의 전체 데이터셋에 대해 학습시킵니다. 한 번의 epoch이란 전체 데이터를 한 번 학습시키는 것을 의미합니다.

# 학습된 가중치 저장
model.save_weights('sample_weight.h5')  # 학습된 가중치를 'sample_weight.h5' 파일로 저장
# `model.save_weights()`는 모델 학습 후 가중치를 저장하는 함수입니다.
# `sample_weight.h5`는 학습된 가중치가 저장될 파일명입니다. 이 파일은 후에 모델을 다시 학습하거나, 다른 곳에서 사용할 때 불러올 수 있습니다.
# `.h5` 확장자는 HDF5 형식으로 데이터를 저장하는 포맷입니다.
