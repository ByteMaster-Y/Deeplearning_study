
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        # learning rate 에 줄인 말
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# SGD(Stochastic Gradient Descent, 확률적 경사 하강법)는 인공지능, 특히 딥러닝에서 많이 사용하는 최적화 알고리즘이에요.

# SGD의 개념
# 경사 하강법(Gradient Descent)의 한 종류로, 전체 데이터가 아니라 **무작위로 선택한 일부 데이터(미니배치 또는 개별 샘플)**를 사용해 가중치를 업데이트하는 방식이에요.

# SGD의 특징
# 빠른 학습: 전체 데이터를 사용하지 않아서 한 번의 업데이트가 빠름
# 노이즈가 많음: 무작위 샘플을 사용하므로 흔들리는 경향이 있음
# 일반화 성능 향상: 노이즈가 오히려 지역 최솟값(local minimum)에 갇히는 것을 방지할 수도 있음
