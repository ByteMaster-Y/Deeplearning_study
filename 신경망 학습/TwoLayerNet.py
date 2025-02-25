import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 아래 코드들 컴파일 결과 해석
# 이 출력은 딥러닝 모델이 훈련(Training)과 검증(Validation) 데이터를 사용하여 학습하는 과정의 결과입니다. 여기서는 10 에폭(epoch)에 걸쳐 훈련과 검증을 수행하고, 각 에폭마다 **손실(loss)**과 **정확도(accuracy)**를 출력하고 있습니다. 구체적으로 무엇을 의미하는지 하나씩 풀어 설명할게요.

# 1. Epoch (에폭)
# 에폭은 모델이 전체 훈련 데이터를 한 번 다 사용한 횟수를 의미합니다. 예를 들어, "Epoch 1/10"이라고 하면, 모델이 훈련 데이터를 한 번 다 사용하고, 그 결과를 출력한 것입니다. 이 과정은 10번 반복되며, 에폭이 끝날 때마다 훈련과 검증의 손실 및 정확도가 업데이트됩니다.

# 2. Loss (손실)
# 손실은 모델의 예측값과 실제 값 간의 차이를 측정하는 지표입니다. 이 값이 낮을수록 모델이 더 잘 예측한다고 볼 수 있습니다. 예를 들어, "Train Loss: 1.6843"은 훈련 데이터에 대한 손실 값이 1.6843이라는 뜻입니다.
# Loss가 감소하는 것이 목표입니다. 즉, 손실 값이 작아질수록 모델이 학습을 잘 하고 있다는 의미입니다.

# 3. Accuracy (정확도)
# 정확도는 모델이 예측한 값과 실제 값이 얼마나 일치하는지를 측정한 비율입니다. 예를 들어, "Train Accuracy: 0.4781"은 훈련 데이터에서 모델이 약 47.81% 정확도로 예측했다는 의미입니다.
# 정확도는 높을수록 좋습니다. 훈련 정확도는 모델이 훈련 데이터에서 얼마나 잘 학습했는지를, 검증 정확도는 학습한 모델이 새로운 데이터에서 얼마나 잘 일반화되는지를 보여줍니다.

# 4. Validation Loss & Validation Accuracy (검증 손실, 검증 정확도)
# 훈련이 끝난 후 모델은 검증 데이터를 사용하여 성능을 평가합니다. 이때 출력되는 Validation Loss와 Validation Accuracy는 훈련 데이터와는 별개의 데이터에서 모델이 얼마나 잘 작동하는지 보여줍니다.
# 예를 들어, "Validation Loss: 0.8241"은 검증 데이터에 대한 손실 값이 0.8241이라는 뜻이고, "Validation Accuracy: 0.8042"는 검증 데이터에 대한 정확도가 80.42%라는 뜻입니다.

# 5. 훈련 및 검증의 변화
# 훈련과 검증의 손실과 정확도는 매 에폭마다 갱신됩니다. 예를 들어:

# Epoch 1/10에서는 훈련 손실이 1.6843이고, 훈련 정확도가 47.81%입니다. 검증 데이터에 대한 손실은 0.8241, 정확도는 80.42%입니다.
# Epoch 10/10에서는 훈련 손실이 0.2348로 크게 감소하고, 훈련 정확도는 93.27%로 증가합니다. 검증 손실은 0.2259로 낮아졌고, 검증 정확도는 93.60%로 증가했습니다.
# 6. 학습 상태
# 훈련 과정 중 손실 값이 점차 감소하고 정확도는 증가하는 것은 모델이 잘 학습하고 있다는 신호입니다. 검증 정확도도 함께 증가하는 경우, 모델이 과적합(overfitting)되지 않고 잘 일반화되고 있다는 뜻입니다.

# 예시로 해석:
# Epoch 1: 모델은 훈련 데이터에서 약 47.8%만 정확히 예측하고, 검증 데이터에서는 80.4% 정확도를 얻었습니다. 손실은 1.6843에서 0.8241로 감소했습니다.
# Epoch 2: 훈련 정확도가 84.6%로 증가하고, 검증 정확도도 88.0%로 향상되었습니다. 손실도 계속해서 감소했습니다.
# Epoch 10: 최종적으로 훈련 정확도는 93.27%, 검증 정확도는 93.60%로 모델의 성능이 크게 향상되었습니다. 손실 값도 훈련 데이터에서 0.2348, 검증 데이터에서 0.2259로 매우 낮아졌습니다.
# 요약:
# 훈련 손실과 정확도는 모델이 훈련 데이터를 얼마나 잘 예측하는지를 나타냅니다.
# 검증 손실과 정확도는 모델이 새로운 데이터(검증 데이터)에서 얼마나 잘 작동하는지를 나타냅니다.
# 학습이 진행될수록 훈련과 검증의 정확도는 증가하고 손실은 감소해야 합니다.

class TwoLayerNet(keras.Model):
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01): # weight_init_std: 가중치 초기화의 표준편차
        super(TwoLayerNet, self).__init__()
        # 두 개의 Dense 레이어를 추가합니다.
        self.dense1 = layers.Dense(hidden_size, input_dim=input_size, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=weight_init_std))
        self.dense2 = layers.Dense(output_size, activation='softmax', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=weight_init_std))

    def call(self, x):
        # 모델의 순전파 함수 정의
        x = self.dense1(x)
        return self.dense2(x)

    def loss(self, x, t):
        # 손실 함수 정의 (교차 엔트로피 손실)
        y = self.call(x)
        return keras.losses.sparse_categorical_crossentropy(t, y)

    def accuracy(self, x, t):
        # 정확도 계산
        y = self.call(x)
        predicted = tf.argmax(y, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, t), tf.float32))
        return accuracy

# 모델 인스턴스화
input_size = 784  # 입력 크기 (MNIST 이미지의 경우 28x28 픽셀)
hidden_size = 50  # 은닉층 뉴런 수
output_size = 10  # 출력 크기 (10개의 클래스)
model = TwoLayerNet(input_size, hidden_size, output_size)

# 옵티마이저와 손실 함수 설정
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# MNIST 데이터셋 로드 및 전처리
(x_train, t_train), (x_test, t_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# 모델 학습
batch_size = 100  # 미니배치 크기
epochs = 10  # 에폭 수
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
