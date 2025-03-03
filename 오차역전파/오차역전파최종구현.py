import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.datasets import mnist

# 데이터 로드 및 전처리 (MNIST 데이터셋 다운로드 및 정규화)
(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train, x_test = x_train.reshape(-1, 784) / 255.0, x_test.reshape(-1, 784) / 255.0  # 픽셀 값을 [0,1] 범위로 정규화
t_train, t_test = keras.utils.to_categorical(t_train, 10), keras.utils.to_categorical(t_test, 10)  # 원-핫 인코딩

# 모델 정의 (입력층-은닉층-출력층 구조, 기존 코드와 동일한 구조로 유지)
model = keras.Sequential([
    layers.Dense(50, activation='relu', input_shape=(784,)),  # 은닉층 (50개 노드, ReLU 활성화 함수)
    layers.Dense(10, activation='softmax')  # 출력층 (10개 클래스, 소프트맥스 활성화 함수)
])

# 모델 컴파일 (손실 함수 및 최적화 방법 설정)
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),  # 확률적 경사 하강법 (SGD) 사용
              loss='categorical_crossentropy',  # 다중 분류 문제를 위한 손실 함수
              metrics=['accuracy'])  # 정확도를 측정 지표로 설정

# 학습 관련 하이퍼파라미터 설정
batch_size = 100  # 미니배치 크기
epochs = 10  # 전체 데이터셋을 10번 반복 학습

# 모델 학습 (fit 메서드를 사용하여 학습 진행)
history = model.fit(x_train, t_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, t_test))

# 모델 평가 (테스트 데이터셋으로 정확도 확인)
loss, acc = model.evaluate(x_test, t_test)
print(f"Test Accuracy: {acc:.4f}")

# 학습 과정에서 저장된 정확도 및 손실 그래프 그리기 (선택사항)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Test Accuracy')
plt.show()
