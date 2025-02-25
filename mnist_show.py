import numpy as np
import tensorflow as tf
from PIL import Image

# MNIST 데이터 로드
(x_train, t_train), (x_test, t_test) = tf.keras.datasets.mnist.load_data()

# 첫 번째 이미지와 레이블
img = x_train[0]
label = t_train[0]
print(label)  # 5
print(img.shape)  # (28, 28)

# 이미지를 PIL 형식으로 변환
img_show = Image.fromarray(np.uint8(img))

# 이미지를 화면에 표시
img_show.show()
