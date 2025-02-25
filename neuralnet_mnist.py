# 필수 라이브러리 임포트
import numpy as np  # 넘파이 라이브러리 (배열 처리 등)
import tensorflow as tf  # 텐서플로우 라이브러리
from tensorflow.keras.models import Sequential  # 모델을 순차적으로 쌓을 수 있도록 하는 모델
from tensorflow.keras.layers import Dense  # 완전 연결층(Dense layer) 정의
from tensorflow.keras.datasets import mnist  # MNIST 데이터셋 로딩
from tensorflow.keras.utils import to_categorical  # 원-핫 인코딩을 위한 유틸리티

# 3.6.1 MNIST 데이터셋 로드
(x_train, t_train), (x_test, t_test) = mnist.load_data()
# x_train, t_train: 학습 데이터 및 레이블
# x_test, t_test: 테스트 데이터 및 레이블

# 데이터 전처리 (평탄화 및 정규화)
# -1은 배열 차원을 변경할 때 나머지 차원의 크기를 자동으로 계산하게 합니다.
# x_train.reshape(x_train.shape[0], -1)에서 -1은 각 28x28 이미지를 784개의 요소를 가진 1차원 배열로 변환하도록 합니다.
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0  # 28x28 크기의 이미지를 1차원 배열로 변환 후, 정규화 (0-255 범위를 0-1로)
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0  # 테스트 데이터도 동일하게 처리

# 레이블을 원-핫 인코딩으로 변환
t_train = to_categorical(t_train, 10)  # 학습용 레이블을 원-핫 인코딩 (10개의 클래스)
t_test = to_categorical(t_test, 10)  # 테스트용 레이블을 원-핫 인코딩 (10개의 클래스)

# 2개의 층을 가진 모델 정의
# 은닉층 128개의 뉴런을 사용하고, 출력층에는 10개의 뉴런을 사용하여 10개의 클래스를 분류

model = Sequential()  # 순차적 모델 선언

# 첫 번째 은닉층 (128개의 뉴런, sigmoid 활성화 함수 사용)
model.add(Dense(128, activation='sigmoid', input_dim=784))  # input_dim=784은 28x28 이미지를 1차원으로 펼쳤을 때의 크기
# 두 번째 층은 출력층으로, 10개의 뉴런을 두고 softmax 활성화 함수를 사용하여 확률로 변환
model.add(Dense(10, activation='softmax'))  # 출력층 (10개의 클래스에 대해 확률값을 출력)

# 학습된 가중치 불러오기
model.load_weights('sample_weight.h5')  # 학습된 가중치를 'sample_weight.h5'에서 불러옴
# .h5 파일은 텐서플로우 모델의 가중치를 저장할 때 사용하는 파일 포맷

# 3.6.3 배치 처리 및 정확도 계산

batch_size = 100  # 한 번에 처리할 데이터 샘플 수 (배치 크기)
accuracy_cnt = 0  # 정확도 계산을 위한 변수 초기화

# 배치 단위로 예측 수행
for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]  # 배치 크기만큼 테스트 데이터를 잘라서 x_batch에 저장
    t_batch = t_test[i:i+batch_size]  # 배치 크기만큼 테스트 레이블을 잘라서 t_batch에 저장
    
    # 예측 수행
    y_batch = model.predict(x_batch)  # x_batch에 대한 예측값을 y_batch에 저장
    p = np.argmax(y_batch, axis=1)  # 예측된 확률값에서 가장 큰 값을 가진 인덱스를 찾아 p에 저장 (예측한 클래스)
    t_true = np.argmax(t_batch, axis=1)  # 실제 레이블에서 가장 큰 값을 가진 인덱스를 찾아 t_true에 저장 (실제 클래스)
    
    accuracy_cnt += np.sum(p == t_true)  # 예측값과 실제값이 일치하는 갯수를 accuracy_cnt에 더함

# 정확도 계산 및 출력
print("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))  # 정확도를 계산하고 출력 (전체 데이터에 대해 정확한 예측의 비율)
