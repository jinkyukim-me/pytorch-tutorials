# imports
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

# For reproducibility
torch.manual_seed(1)

# 데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]] # |x_data|=(6,2) m=6 d=2
y_data = [[0], [0], [0], [1], [1], [1]] # |y_data|=(6,1)
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
# print(x_train.shape)
# print(y_train.shape)

# 모델 초기화
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.05)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b) # or .mm or @
    cost = -(y_train * torch.log(hypothesis) +
             (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 10번마다 로그 출력
    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# 모델의 정확도 계산
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
prediction = hypothesis >= torch.FloatTensor([0.5])
correct_prediction = prediction.float() == y_train
accuracy = correct_prediction.sum().item() / len(correct_prediction)
print('The model has an accuracy of {:2.2f}% for the training set.'
      .format(accuracy * 100))