
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        # learning rate 에 줄인 말
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

