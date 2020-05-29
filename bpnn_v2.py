# -*- encoding: utf-8 -*-
'''
@Time    :   2020/04/26 10:59:38
@Author  :   Haoran Li 
@Contact :   lihr16@163.com
@Desc    :   BPNN model v2: 3 hidden layer
'''
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#超参数
EPOCH=100
BATCH_SIZE=100
ACFs=[150,175,25,150,125,175]
ycols=[19,20,21,22,23,24]
obs=4
ACF=ACFs[obs]
ycol=ycols[obs]
# 数据集,其中dataraw未经过归一化，datanorm经过归一化
dataraw=pd.read_csv("Data_8000_nn.csv")
dataraw=np.array(dataraw,dtype=np.float32)
datanorm=pd.read_csv("Data_8000_normalized.csv")
datanorm=np.array(datanorm,dtype=np.float32)
#训练集
x=datanorm[200:5200,10:19]#OV
y=dataraw[240:5240,ycol]
hist=datanorm[240-ACF:5240-ACF,ycol]
hist=np.expand_dims(hist,axis=1)#SV历史值
x=np.hstack((x,hist))
x=torch.tensor(x)
y=torch.unsqueeze(torch.tensor(y),dim=1)
#测试集
test_x=datanorm[5200:6200,10:19]
test_y=dataraw[5240:6240,ycol]
test_hist=datanorm[5240-ACF:6240-ACF,ycol]
test_hist=np.expand_dims(test_hist,axis=1)
test_x=np.hstack((test_x,test_hist))
test_x=torch.tensor(test_x)
test_y=torch.unsqueeze(torch.tensor(test_y),dim=1)


# Data Loader for easy mini-batch return in training
train_data=Data.TensorDataset(x, y)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)



# define the network
net = torch.nn.Sequential(
    torch.nn.Linear(10,60),
    torch.nn.Linear(60,40),
    torch.nn.Linear(40,20),
    torch.nn.Linear(20,1)
)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_func = torch.nn.L1Loss()  
losses=[]

for epoch in range(EPOCH):
    for step, (b_x,b_y) in enumerate(train_loader):
        prediction = net(b_x)    
        loss = loss_func(prediction, b_y)     
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    losses.append(loss.data.numpy())

# plot train result
plt.title('End#Distance train')
trainindex=np.array(range(EPOCH))
plt.plot(trainindex, losses,'r-',lw=2)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.text(50, 0.5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
plt.savefig('./bpnn_v2_5_train.png')
plt.show()

# print('Loss=%.4f' % loss.data.numpy())
torch.save(net, 'bpnn_v2_5.pkl') 
# test
net2 = torch.load('bpnn_v2_5.pkl')
prediction = net2(test_x)
test_loss=loss_func(prediction, test_y)
print(test_loss.data.numpy())

# plot test result
plt.title('End#Distance test')
testindex=np.array(range(1000))
plt.plot(testindex, test_y.data.numpy(),'b-',lw=2)
plt.plot(testindex, prediction.data.numpy(), 'r-', lw=2)
plt.xlabel('sample')
plt.ylabel('End#Distance (m)')
plt.legend(["true","predicted"])
plt.savefig('./bpnn_v2_5_test.png')
plt.show()