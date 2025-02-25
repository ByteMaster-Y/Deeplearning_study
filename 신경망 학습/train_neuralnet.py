import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# 오버피팅 막는 방법: 가중치 감소, 드롭아웃

# MNIST 데이터셋 로드 (정답 레이블을 one-hot 인코딩 하지 않음)
(x_train, t_train), (x_test, t_test) = mnist.load_data()

# 데이터 전처리
x_train = x_train.reshape(-1, 28*28) / 255.0  # 28x28 이미지를 1차원 배열로 변경 후 정규화
x_test = x_test.reshape(-1, 28*28) / 255.0

# 모델 정의
model = keras.Sequential([
    layers.Dense(50, input_dim=784, activation='relu'),  # 784 -> 50개의 뉴런
    layers.Dense(10, activation='softmax')  # 50 -> 10개의 출력 (10개의 클래스)
])

# 손실 함수 및 옵티마이저 설정
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# 하이퍼파라미터 설정
batch_size = 100  # 미니배치 크기
epochs = 10  # 에폭 수

# 모델 학습 (미니배치 학습 포함)
history = model.fit(x_train, t_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, t_test))

# 학습 결과 출력
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss[epoch]}, Train Accuracy: {train_acc[epoch]}")
    print(f"Validation Loss: {val_loss[epoch]}, Validation Accuracy: {val_acc[epoch]}")
