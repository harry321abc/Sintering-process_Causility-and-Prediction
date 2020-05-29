%9��OV��6��SV��������ʱ�ͣ�������������Ӱ�죬ʹ��xmap��ֻ��11-19��20-25�ű���

% Read the normalized data
filename='D:\���Ʊ���\ccm-m\Data_8000_normalized.csv';
Data_norm=csvread(filename,0,0,[0,0,7999,24]);
N_data = 8000; % length of the time series to be generated

% Parameters for phase space emdbedding
tau = 1; % Time delay
E = 3;   % Embedding dimension

Lmin = 1000;
Lmax = N_data-(E-1)*tau; % Max possible Library size
NL = 40;
Lstep = round((Lmax-Lmin)/(NL-1));
LibLen = [Lmin:Lstep:Lmax]';
NL = numel(LibLen);
rho= zeros(9,6,NL);

% Loop over Library length
%
for i=11:19
    X=Data_norm(:,i);
    X=X';
    for j=20:25
        Y=Data_norm(:,j);
        Y=Y';
        MX = psembed(X,E,tau);
        MY = psembed(Y,E,tau);
        for l=1:NL
            L=LibLen(l);
            M = round(Lmax/L);
            rho_M = zeros(M,1);
            for r = 1:M
                [ X_MY, Y_MX, X1, Y1] = ...
                    xmap( X, Y, MX, MY, E, tau, L,'random');
                rho_M(r) = corr(X_MY,X1');
            end
            rho(i-10,j-19,l) = mean(rho_M);
        end
    end
end

