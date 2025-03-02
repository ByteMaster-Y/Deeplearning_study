# 오차역전파는 Backpropagation이야.

# 간단히 말하면, 신경망 학습에서 **오차(손실)**를 출력층에서 입력층 방향으로 거꾸로 전파하면서 가중치를 조정하는 알고리즘이야. 
# 이를 통해 신경망이 더 정확한 예측을 할 수 있도록 학습해! 😊

# 5.4.1 곱셈 계층
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


# 5.4.2 덧셈 계층
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


if __name__ == '__main__':
    # 문제1의 예시
    apple = 100
    apple_num = 2
    tax = 1.1

    # 계층들
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # 순전파
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)

    print(price)  # 220.0

    # 역전파
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(dapple, dapple_num, dtax)  # 2.2 110.0 200
    # dapple = 2.2 → 사과 가격(100)을 변경했을 때, 최종 가격에 미치는 영향
    # dapple_num = 110 → 사과 개수(2)를 변경했을 때, 최종 가격에 미치는 영향
    # dtax = 200 → 세금(1.1)을 변경했을 때, 최종 가격에 미치는 영향

    # 문제2의 예시
    orange = 150
    orange_num = 3

    # 계층들
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    # 순전파
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)

    print(price)  # 715.0

    # 역전파
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dornage, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(dapple_num, dapple, dornage, dorange_num, dtax)
    # 110.0 2.2 3.3 165.0 650