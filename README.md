# Sintering-process_Causility-and-Prediction
  This repository contains the project related to the sintering process in iron-making industry, in which the author conducts causility analysis and state variables prediction.
# Denoising and normalization
  3sigma cut off and moving window average method are used in denoising and normalization, which contain in ***Pretreat.m***. 
## Causality
  This project uses two kind of causality test: Autocorrelation Function(ACF) and Convergent cross-mapping(CCM), which correspond to ***acf.py*** and ***ccm_v8.m*** and ***ccm_v9***, respectivly.
## ANN Prediction
  Commoly used ANN with fully connected layers are adopted in this project. However, due to difference in hidden layers and input dimension, four versions of BPNN are developed. You can find them as ***BPNN_v1***,***BPNN_v2***,***BPNN_v3*** and ***BPNN_v4***.
## KDE threshold
  In order to find the stationary threshold for system state variables, the kernal density estimation(KDE) with the Gaussian kernel is employed. Run the ***kde.m*** and you will get the probablity distribution.
  
