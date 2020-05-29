# -*- encoding: utf-8 -*-
'''
@Time    :   2020/04/23 10:15:39
@Author  :   Haoran Li 
@Contact :   lihr16@163.com
@Desc    :   BPNN model v3: 3 hidden layer; Based on window input, window size=3
'''

import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load_data_window import x,y,test_x,test_y

# 数据集,其中dataraw未经过归一化，datanorm经过归一化
dataraw=pd.read_csv("Data_8000_nn.csv")
dataraw=np.array(dataraw,dtype=np.float32)
datanorm=pd.read_csv("Data_8000_normalized.csv")
datanorm=np.array(datanorm,dtype=np.float32)
#取值参数
ACFs=[150,175,25,150,125,175]
ycols=[19,20,21,22,23,24]
lag1=40
lag2=30
lag3=20
obs=4
ACF=ACFs[obs]
ycol=ycols[obs]

#训练集
#操作变量序列
x1=datanorm[240-lag1:5240-lag1,10:19]#OV
x2=datanorm[240-lag2:5240-lag2,10:19]
x3=datanorm[240-lag3:5240-lag3,10:19]
#历史值序列
hist1=datanorm[240-ACF:5240-ACF,ycol]
hist2=datanorm[240-2*ACF//3:5240-2*ACF//3,ycol]
hist3=datanorm[240-ACF//3:5240-ACF//3,ycol]
hist1=np.expand_dims(hist1,axis=1)#SV历史值
hist2=np.expand_dims(hist2,axis=1)
hist3=np.expand_dims(hist3,axis=1)
#合并序列
x=np.hstack((x1,x2,x3,hist1,hist2,hist3))
y=dataraw[240:5240,ycol]
x=torch.tensor(x)
y=torch.unsqueeze(torch.tensor(y),dim=1)

#测试集
#操作变量序列
test_x1=datanorm[5240-lag1:6240-lag1,10:19]
test_x2=datanorm[5240-lag2:6240-lag2,10:19]
test_x3=datanorm[5240-lag3:6240-lag3,10:19]
#历史值序列
test_hist1=datanorm[5240-ACF:6240-ACF,ycol]
test_hist2=datanorm[5240-2*ACF//3:6240-2*ACF//3,ycol]
test_hist3=datanorm[5240-ACF//3:6240-ACF//3,ycol]
test_hist1=np.expand_dims(test_hist1,axis=1)
test_hist2=np.expand_dims(test_hist2,axis=1)
test_hist3=np.expand_dims(test_hist3,axis=1)
#合并序列
test_x=np.hstack((test_x1,test_x2,test_x3,test_hist1,test_hist2,test_hist3))
test_y=dataraw[5240:6240,ycol]
test_x=torch.tensor(test_x)
test_y=torch.unsqueeze(torch.tensor(test_y),dim=1)

#超参数
EPOCH=100
BATCH_SIZE=100

# Data Loader for easy mini-batch return in training
train_data=Data.TensorDataset(x, y)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# define the network
net = torch.nn.Sequential(
    torch.nn.Linear(30,60),
    torch.nn.Linear(60,40),
    torch.nn.Linear(40,20),
    torch.nn.Linear(20,1)
)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_func = torch.nn.L1Loss()  #平均绝对误差
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
plt.savefig('./bpnn_v3_5_train.png')
plt.show()

# print('Loss=%.4f' % loss.data.numpy())
torch.save(net, 'bpnn_v3_5.pkl') 
# test
net2 = torch.load('bpnn_v3_5.pkl')
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
plt.savefig('./bpnn_v3_5_test.png')
plt.show()