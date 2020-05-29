%9个OV对6个SV，考虑时滞，使用crossmap，只算11-19对20-25号变量，时滞取-40:5:40，中期答辩用
% Read the normalized data
filename='D:\ccm-m\Data_8000_normalized.csv';
Data_norm=csvread(filename,0,0,[0,0,7999,24]);
% Choose parameters
rho=zeros(9,6,17);
E = 3;
tau=1;
N=8000;% slice length
% Loop over i,j
% i:X index, j:Y index
for i=11:19
    X=Data_norm(:,i);
    X=X';
    for j=20:25
        Y=Data_norm(:,j);
        Y=Y';
        MX = psembed(X,E,tau);
        MY = psembed(Y,E,tau);
        L = 5000; 
        for m=-40:5:40
            [ X_MY, Y_MX, X1, Y1] = crossmap( X, Y, MX, MY, E, tau, L,'linear',m);
            rho(i-10,j-19,m/5+9)=corr(X_MY,X1');
        end
    end
end