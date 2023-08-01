import numpy as np 
import dezero
from dezero import optimizers
import dezero.functions as F 
from dezero.models import MLP 
from dezero.datasets import Spiral
from dezero.dataloaders import DataLoader

max_epoch = 500 
batch_size = 30 
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0 

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)

        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    
    avg_loss = sum_loss / len(train_set)
    avg_acc = sum_acc / len(train_set)
    print('epoch: {}'.format(epoch+1))
    print('train loss: {:4f}, accuracy: {:4f}'.format(avg_loss, avg_acc))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy_simple(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    avg_loss = sum_loss / len(test_set)
    avg_acc = sum_acc / len(test_set)
    print('test loss: {:4f}, accuracy: {:4f}'.format(avg_loss, avg_acc))