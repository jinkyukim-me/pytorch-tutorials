import torch
import torch.optim as optim
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w,b], lr=0.038)

nb_epochs = 1000
for epoch in range(1, nb_epochs + 1):
    hypothesis = x_train * w + b
    loss = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 10번마다 로그 출력
    if epoch % 10 == 0:
        print('Epoch {:4d}/{} W: {:.8f}, b: {:.8f} Cost: {:.8f}'
            .format(epoch, nb_epochs, w.item(), b.item(), loss.item()))
