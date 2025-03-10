import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

"""

# CNN을 이용한 MNIST 숫자 분류

이 프로젝트는 **CNN(Convolutional Neural Network)**을 사용하여 MNIST 숫자 데이터를 분류하는 모델을 구현한 것입니다.

## 1. 데이터 로드 및 전처리

```python
(x_train, t_train), (x_test, t_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 정규화
x_train = x_train[..., np.newaxis]  # (60000, 28, 28, 1) 형태로 변환
x_test = x_test[..., np.newaxis]
```

### 📌 핵심 포인트
- **MNIST 데이터셋 로드**: 28x28 크기의 손글씨 숫자 이미지.
- **정규화**: 픽셀 값을 0~1 범위로 변환하여 학습 성능 향상.
- **차원 변환**: CNN 모델이 4D 입력(batch, height, width, channel)을 요구하므로 `(28,28)`을 `(28,28,1)`로 변환.

## 2. CNN 모델 구조

```python
model = keras.Sequential([
    layers.Conv2D(30, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### 📌 핵심 포인트
1. **Conv2D(30, (5,5), activation='relu')**
   - 30개의 5×5 필터 적용하여 특징 추출.
   - ReLU 활성화 함수 사용.
2. **MaxPooling2D((2,2))**
   - 2×2 풀링을 적용하여 특징 맵 크기 절반 감소.
3. **Flatten()**
   - CNN 출력을 1차원 벡터로 변환.
4. **Dense(100, activation='relu')**
   - 100개의 뉴런을 가진 완전연결층.
5. **Dense(10, activation='softmax')**
   - 10개의 클래스(숫자 0~9)를 예측하는 출력층.

## 3. 모델 컴파일 및 학습

```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

```python
history = model.fit(x_train[:5000], t_train[:5000], epochs=20, batch_size=100,
                    validation_data=(x_test[:1000], t_test[:1000]))
```

### 📌 핵심 포인트
- **Adam 옵티마이저 사용**: 적응형 학습률 조정.
- **손실 함수**: `sparse_categorical_crossentropy` (정답이 원-핫 인코딩이 아님).
- **배치 크기(batch_size=100)**: 메모리와 성능 균형 조절.
- **검증 데이터 설정**: `validation_data=(x_test[:1000], t_test[:1000])`로 학습 중 성능 확인.

## 4. 학습 결과 시각화

```python
plt.plot(history.history['accuracy'], marker='o', label='train', markevery=2)
plt.plot(history.history['val_accuracy'], marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
```

### 📌 핵심 포인트
- 학습 정확도(`accuracy`)와 검증 정확도(`val_accuracy`)를 비교하여 **과적합 여부 판단**.
- 훈련 데이터와 검증 데이터의 차이가 크면 **과적합 가능성**이 있음.

## 5. 최종 평가

```python
final_acc = model.evaluate(x_test[:1000], t_test[:1000], verbose=0)[1]
print(f"Final Test Accuracy: {final_acc:.3f}")
```

### 📌 핵심 포인트
- `evaluate()`를 사용하여 모델의 일반화 성능을 확인.
- 테스트 정확도가 **95% 이상이면 성공적인 모델**.

---

## 🔥 집중해서 봐야 할 부분

### ✅ 데이터 전처리
- `np.newaxis`로 CNN 입력 형태 맞추기.
- 픽셀 값 정규화(0~1).

### ✅ CNN 구조
- `Conv2D` 필터 개수, 크기 조정이 모델 성능에 영향.
- `MaxPooling2D`를 사용해 연산량 감소.
- 마지막 `Dense(10, softmax)`에서 클래스 확률 예측.

### ✅ 학습 과정 및 결과 분석
- `batch_size`, `epochs` 조절하여 최적화.
- 학습/검증 정확도 비교로 과적합 여부 확인.

### ✅ 최종 모델 평가
- `evaluate()`를 통해 일반화 성능 확인.
- 정확도가 낮다면 **모델 구조 조정, 데이터 증강, 하이퍼파라미터 튜닝** 필요.

---

이제 CNN을 활용한 MNIST 분류 모델을 이해하고, 실험해볼 준비가 되었습니다! 🚀

"""


# 데이터 로드 및 전처리
(x_train, t_train), (x_test, t_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 정규화
x_train = x_train[..., np.newaxis]  # (60000, 28, 28, 1) 형태로 변환
x_test = x_test[..., np.newaxis]

# 모델 생성
model = keras.Sequential([
    layers.Conv2D(30, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(x_train[:5000], t_train[:5000], epochs=20, batch_size=100,
                    validation_data=(x_test[:1000], t_test[:1000]))

# 학습 결과 그래프 그리기
plt.plot(history.history['accuracy'], marker='o', label='train', markevery=2)
plt.plot(history.history['val_accuracy'], marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# 최종 테스트 정확도 출력
final_acc = model.evaluate(x_test[:1000], t_test[:1000], verbose=0)[1]
print(f"Final Test Accuracy: {final_acc:.3f}")
