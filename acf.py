import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import *
# load data
data=pd.read_csv("Data_8000_normalized.csv")
data=np.array(data,dtype=np.float32)
x1=data[1:6000,19]
x2=data[1:6000,20]
x3=data[1:6000,21]
x4=data[1:6000,22]
x5=data[1:6000,23]
x6=data[1:6000,24]

plot_acf(x1,use_vlines=True,lags=200,title='ACF 14#Pres')
plt.show()
plot_acf(x2,use_vlines=True,lags=200,title='ACF 22#Pres')
plt.show()
plot_acf(x3,use_vlines=True,lags=200,title='ACF 14#Temp')
plt.show()
plot_acf(x4,use_vlines=True,lags=200,title='ACF 22#Temp')
plt.show()
plot_acf(x5,use_vlines=True,lags=200,title='ACF DistEnd')
plt.show()
plot_acf(x6,use_vlines=True,lags=200,title='ACF TempEnd')
plt.show()
#print(stattools.adfuller(x1)) 
