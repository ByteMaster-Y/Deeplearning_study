import numpy as np

# a = np.array([0.3,2.9,4.0])
# exp_a = np.exp(a) # 지수 함수
# print(exp_a)
# sum_exp_a = np.sum(exp_a)
# print(sum_exp_a)
# y = exp_a / sum_exp_a
# print(y)

# def softmax(a):
    # exp_a = np.exp(a)
    # sum_exp_a = np.sum(exp_a)
    # y = exp_a / sum_exp_a

    # return y

# 오버플로 대체
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# 소프트맥스 함수의 출력은 0-1.0 사이의 실수. 또 소프트맥스 함수의 출력의 총합은1이다.
# 출력 총합이 1이 된다는 점은 굉장히 중요한 성질이다.
a = np.array([0.3,2.9,4.0])
y = softmax(a)
print(y)
# [0.01821127 0.24519181 0.73659691] 이 출력의 이미를 보면, 2번째 인덱스의 확률이 73퍼센트이기에 답은 2번째 클래스라고 정할 수 있다.
# 소프트 맥스 함수를 적용해도 각 원소의 대소 관계는 변화지 않는다. 이는 지수함수가 단조 증가 함수 이기 때문입니다.
# 일반적으로 신경망을 학습시킬 때는 출력층에서 소프트맥스 함수를 사용한다.


