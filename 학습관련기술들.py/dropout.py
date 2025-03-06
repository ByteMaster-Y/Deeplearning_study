"""
오버피팅은 주로 다음의 경우에 일어난다.
 * 매개변수가 많고 표현력이 높은 모델
 * 훈련 데이터가 적음

"""


# 오버피팅 막는 방법
# 1) 가중치 감소
# 2) 드롭아웃 사용

import numpy as np

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        """
        드롭아웃(Dropout) 클래스  
        - dropout_ratio: 드롭아웃 비율 (기본값 0.5)
        - mask: 활성 뉴런을 결정하는 마스크
        """
        self.dropout_ratio = dropout_ratio
        self.mask = None  # 학습 중 활성화할 뉴런을 저장하는 마스크

    def forward(self, x, train_flg=True):
        """
        순전파 (Forward)  
        - 학습 중(train_flg=True)일 때: 뉴런을 랜덤으로 비활성화  
        - 테스트 중(train_flg=False)일 때: 드롭아웃 비율을 반영해 출력값 조정
        """
        if train_flg:
            # 입력과 같은 shape의 랜덤 행렬 생성 (dropout_ratio보다 크면 1, 작으면 0)
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio  
            return x * self.mask  # 선택된 뉴런만 활성화
        else:
            return x * (1.0 - self.dropout_ratio)  # 테스트 시 드롭아웃 효과 보정

    def backward(self, dout):
        """
        역전파 (Backward)  
        - 드롭된 뉴런은 0이므로, 미분값도 해당 부분은 0으로 유지
        """
        return dout * self.mask if self.mask is not None else dout  # mask 적용

# 테스트 코드

# 입력 데이터 생성 (3x3 행렬)
x = np.array([[1.0, 2.0, 3.0], 
              [4.0, 5.0, 6.0], 
              [7.0, 8.0, 9.0]])

dropout = Dropout(0.5)  # 드롭아웃 비율 50%

# 학습 중 (train_flg=True)
train_output = dropout.forward(x, train_flg=True)
print("Train Output:\n", train_output)

# 테스트 중 (train_flg=False)
test_output = dropout.forward(x, train_flg=False)
print("Test Output:\n", test_output)

# 역전파 테스트
dout = np.ones_like(x)  # 모두 1인 미분값
backprop_output = dropout.backward(dout)
print("Backward Output:\n", backprop_output)

"""

학습 중(Train Output)에서 드롭아웃 효과 확인

Train Output:
 [[1. 2. 0.]
  [4. 0. 0.]
  [7. 8. 0.]]
입력 데이터 x에서 일부 값이 0이 되었어요.
드롭아웃 확률이 50%이므로, 약 절반 정도의 뉴런이 비활성화되었어요.
예를 들어, (1,1), (1,2), (2,1) 등은 살아남았고, (1,3), (2,2), (2,3) 등은 0이 되었어요.
이 과정에서 랜덤한 뉴런을 끄는(drop) 역할을 한다는 걸 확인할 수 있어요.

2. 테스트 중(Test Output)에서 스케일 조정 확인

Test Output:
 [[0.5 1.  1.5]
  [2.  2.5 3. ]
  [3.5 4.  4.5]]
학습 중에는 뉴런을 일부 끄지만, 테스트할 때는 모든 뉴런을 사용해야 해요.
그래서 테스트 단계에서는 출력값을 (1 - dropout_ratio) 만큼 곱해서 보정해요.
여기서는 0.5를 곱했어요.
예를 들어, 원래 입력값이 1.0이었다면 1.0 * 0.5 = 0.5가 된 거예요.

3. 역전파(Backward Output)에서 영향 확인
Backward Output:
 [[1. 1. 0.]
  [1. 0. 0.]
  [1. 1. 0.]]
순전파 때 0이 된 뉴런은 역전파에서도 0으로 유지돼요.
뉴런이 비활성화되었으니, 해당 뉴런에 대한 미분 값도 계산하지 않는 거죠.
예를 들어, (1,3), (2,2), (2,3), (3,3) 위치는 순전파 때 0이었고, 역전파에서도 0이 유지되었어요.

🔍 결론: 테스트를 통해 확인한 점
✅ 학습 중에는 일부 뉴런을 랜덤으로 끄는(drop) 과정이 적용됨
✅ 테스트할 때는 출력값을 보정해서 전체 뉴런을 활용할 수 있도록 함
✅ 역전파 과정에서도 비활성화된 뉴런은 그대로 0이 유지됨

드롭아웃이 어떻게 학습 중 과적합을 방지하는지 직접 확인할 수 있는 좋은 테스트였어요! 🚀

"""