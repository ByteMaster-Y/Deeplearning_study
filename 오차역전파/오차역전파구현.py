import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np

# 1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # MNIST 데이터셋 불러오기

# 이미지 데이터를 1차원 벡터로 변환하고 정규화 (0~255 → 0~1)
x_train, x_test = x_train.reshape(-1, 784) / 255.0, x_test.reshape(-1, 784) / 255.0  

# 정답 레이블을 원-핫 인코딩 형태로 변환 (숫자 레이블 → 10차원 벡터)
y_train, y_test = keras.utils.to_categorical(y_train, 10), keras.utils.to_categorical(y_test, 10)

# 2. 모델 생성 (Sequential API 사용)
model = keras.Sequential([
    layers.Dense(50, activation='relu', input_shape=(784,)),  # 첫 번째 은닉층 (50개 뉴런, ReLU 활성화 함수)
    layers.Dense(10, activation='softmax')  # 출력층 (10개 뉴런, 소프트맥스 활성화 함수)
])

# 3. 모델 컴파일
model.compile(
    optimizer='sgd',  # 확률적 경사 하강법(SGD) 사용
    loss='categorical_crossentropy',  # 다중 클래스 분류를 위한 손실 함수
    metrics=['accuracy']  # 모델 평가 지표로 정확도 사용
)

# 4. 모델 훈련
model.fit(
    x_train, y_train,  # 훈련 데이터
    epochs=10,  # 전체 데이터셋을 10번 반복 학습 (10 에포크)
    batch_size=32,  # 미니배치 크기 (한 번에 32개 샘플씩 학습)
    validation_data=(x_test, y_test)  # 검증 데이터로 성능 평가
)

# 5. 모델 평가 (테스트 데이터셋에서 성능 측정)
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.4f}")  # 테스트 정확도 출력

# 6. 기울기(Gradient) 계산 (GradientTape 사용)
def compute_gradients(model, x_batch, y_batch):
    with tf.GradientTape() as tape:  # 자동 미분을 위한 GradientTape 사용
        loss = model.compiled_loss(y_batch, model(x_batch, training=True))  # 손실 함수 계산
    grads = tape.gradient(loss, model.trainable_variables)  # 모델의 학습 가능한 변수에 대한 기울기 계산
    return grads

# 7. 일부 데이터에 대한 기울기 확인
x_batch = tf.convert_to_tensor(x_train[:3], dtype=tf.float32)
y_batch = tf.convert_to_tensor(y_train[:3], dtype=tf.float32)

grads = compute_gradients(model, x_batch, y_batch)  # 기울기 계산

# 8. 각 가중치 변수의 기울기 출력
for var, grad in zip(model.trainable_variables, grads):
    print(f"{var.name}: {np.mean(np.abs(grad.numpy()))}")  # 기울기의 절댓값 평균 출력
