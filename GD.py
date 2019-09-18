import torch
# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
W = torch.zeros(1)
# Learning rate 설정
lr = 0.01

nb_epochs = 100
for epoch in range(1, nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W

    # cost gradient 계산
    cost = torch.mean((hypothesis - y_train)**2)
    gradient = torch.sum((W * x_train - y_train) * x_train)

    print('Epoch {:4d}/{} W: {:.8f} Cost: {:8f}'
          .format(epoch, nb_epochs, W.item(), cost.item()))

    # cost gradient로 H(x) 개선
    W = W - lr * gradient
