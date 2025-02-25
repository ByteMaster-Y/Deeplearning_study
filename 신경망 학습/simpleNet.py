import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 🔍 코드 상세 설명

# 📌 1. 신경망 정의 (model)
# Dense(3, input_shape=(2,), activation=None)
# 입력 크기: 2 (특징 2개)
# 출력 크기: 3 (클래스 3개)
# activation=None: 활성화 함수 없음 (소프트맥스 적용 전 로짓 값 반환)

# 📌 2. 손실 함수 및 옵티마이저 설정 (compile())
# SGD (Stochastic Gradient Descent): 확률적 경사 하강법
# 학습률 0.1로 설정
# gradient descent를 사용해 가중치를 갱신할 예정
# CategoricalCrossentropy (from_logits=True)
# from_logits=True이므로, Dense의 출력값(logits)에 직접 적용
# 내부적으로 softmax + cross entropy를 자동으로 처리함

# 📌 3. 입력값 & 정답 레이블 (x, t)
# x = np.array([[0.6, 0.9]])
# 입력 데이터 2개 (특징이 2개인 샘플)
# t = np.array([[0, 0, 1]])
# 정답이 index 2인 one-hot encoding 레이블

# 📌 4. 예측값 계산 (model.predict(x))
# model.predict(x)는 Dense 층의 선형 변환 결과 로짓(logits) 을 반환
# np.argmax(p): 가장 큰 값을 가진 인덱스(즉, 예측한 클래스)

# 📌 5. 손실 계산 (model.evaluate(x, t))
# 현재 가중치 상태에서 손실값(오차) 를 계산
# evaluate() 함수는 predict() + loss 계산을 한번에 수행

# 📌 6. 기울기(gradient) 계산 (GradientTape)
# tf.GradientTape() 사용
# GradientTape는 연산 과정을 기록하여 자동 미분을 가능하게 함
# 순전파(logits = model(x, training=True)) 수행 후, 손실값 계산
# tape.gradient(loss_value, model.trainable_variables) 호출 시 가중치에 대한 기울기 자동 계산

# 🔥 결론: 이 코드가 하는 일
# Dense(2 → 3)로 이루어진 신경망 생성
# 입력값 x = [0.6, 0.9]을 받아 예측값(logits) 계산
# 정답 t = [0, 0, 1]과 비교하여 손실값 측정
# GradientTape로 손실을 가중치 W에 대해 미분 (기울기 계산)
# 이후 SGD를 사용해 가중치 갱신 가능 (현재 코드는 학습 과정 포함 X)

# 1. 신경망 모델 정의
model = keras.Sequential([
    layers.Dense(3, input_shape=(2,), activation=None)  # 입력 크기 2, 출력 크기 3
    # activation=None -> 소프트맥스 적용 전, 로짓(logits) 값 출력
])

# 2. 손실 함수 및 옵티마이저 설정
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.1),  # 경사 하강법(Stochastic Gradient Descent)
    loss=keras.losses.CategoricalCrossentropy(from_logits=True)  
    # 소프트맥스 + 크로스엔트로피 손실 함수 (from_logits=True 설정)
)

# 3. 임의의 입력값 (예제 데이터)
x = np.array([[0.6, 0.9]])  # 입력 데이터 (2개의 특징)
t = np.array([[0, 0, 1]])  # 정답 레이블 (one-hot encoding, 정답은 index 2)

# 4. 예측값 계산
p = model.predict(x)  # 신경망이 예측한 로짓(logits) 값
print("예측값 (logits):", p)
print("최댓값 인덱스 (예측 클래스):", np.argmax(p))  # 가장 높은 확률을 가진 클래스 출력

# 5. 손실 계산 (현재 가중치 상태에서 손실값 확인)
loss = model.evaluate(x, t, verbose=0)  # 손실 값 계산 (예측값과 정답 비교)
print("손실 값:", loss)

# with 문은 파이썬에서 특정 블록의 실행을 관리하는 기능이야.
# 특히 TensorFlow에서 with tf.GradientTape()는 자동으로 미분을 계산하는 역할을 해!
# 6. 기울기 계산 (GradientTape 사용)
with tf.GradientTape() as tape:
    logits = model(x, training=True)  # 순전파(forward propagation) 수행 (logits 계산)
    loss_value = keras.losses.categorical_crossentropy(t, logits, from_logits=True)
    # 손실 함수 계산 (소프트맥스 적용 없이 로짓 값을 바로 입력)

# 7. 기울기(gradient) 계산: 손실 값을 가중치 W에 대해 미분
grads = tape.gradient(loss_value, model.trainable_variables)
print("기울기 (가중치 W에 대한 미분값):", grads[0].numpy())  # 첫 번째 가중치에 대한 기울기 출력
