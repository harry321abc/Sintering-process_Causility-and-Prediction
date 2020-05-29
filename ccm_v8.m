%9��OV��6��SV������ʱ�ͣ�ʹ��crossmap��ֻ��11-19��20-25�ű�����ʱ��ȡ-40:5:40�����ڴ����
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