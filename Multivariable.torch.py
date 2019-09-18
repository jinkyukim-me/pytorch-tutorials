import torch
import torch.optim as optim
# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# 모델 초기화
W = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.00001)

nb_epochs = 20000
for epoch in range(1, nb_epochs + 1):
    # H(x) 계산
    hypothesis = x_train.matmul(W) + b # or .mm or @
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    # 10마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'
              .format(epoch, nb_epochs, hypothesis.squeeze().detach(),
                      cost.item()))

print(y_train)
