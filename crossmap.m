function [ X_MY, Y_MX, X1, Y1] = crossmap( X, Y, MX, MY, E, tau, L, sampling,lag)
    [n, ex] = size(MX);
    [n, ey] = size(MY);
%
% Extract a library of 'L' points from MX and MY
%
switch (sampling)
    case 'linear'
        MX = MX(40:L+39,:);
        MY = MY(40:L+39,:);
    case 'random'
        for var = 1:2
            N_indices = 2*L;
            sample_indices = unique(randi(n,[N_indices 1]));
            while numel(sample_indices) < L
                N_indices = round(1.2 * N_indices); % Increase by 20%
                sample_indices = unique(randi(n,[N_indices 1]));
            end

            sample_indices = sample_indices(randperm(numel(sample_indices)));
            switch (var)
                case 1
                    idx = sample_indices(1:L);
                    MX = MX(idx,:);
                case 2
                    idy = sample_indices(1:L);
                    MY = MY(idy,:);
            end
        end
    otherwise
        errorstring = sprintf('Unknown sampling type: %s',sampling);
        error('xmap:sampling', errorstring);
end

[n,d]=knnsearch(MX,MX,'k',E+2,'distance','euclidean');

switch (sampling)
    case 'linear'
        data_nn = Y(n(:,2:E+2)+(E-1)*tau);  
    case 'random'
        data_nn = Y(idx(n(:,2:E+2))+(E-1)*tau);  
end

EPS = 1e-8;
w = zeros(L,E+1);
for p=1:L
    w(p,:) = exp(-d(p,2:E+2)/(EPS+d(p,2)));
    w(p,:) = w(p,:)/sum(w(p,:));
end
Y_MX = w .* data_nn;
Y_MX = sum(Y_MX,2);

[n,d]=knnsearch(MY,MY,'k',E+2,'distance','euclidean');
switch (sampling)
    case 'linear'
        data_nn = X(n(:,2:E+2)+(E-1)*tau);
    case 'random'
        data_nn = X(idy(n(:,2:E+2))+(E-1)*tau);
end
for p=1:L
    w(p,:) = exp(-d(p,2:E+2)/(EPS+d(p,2)));
    w(p,:) = w(p,:)/sum(w(p,:));
end
X_MY = w .* data_nn;
X_MY = sum(X_MY,2);

switch (sampling)
    case 'linear'
        X1 = X(40+(E-1)*tau+lag:L+39+(E-1)*tau+lag);
        Y1 = Y(40+(E-1)*tau+lag:L+39+(E-1)*tau+lag);
    case 'random'
        X1 = X(idy+(E-1)*tau+lag);
        Y1 = Y(idx+(E-1)*tau+lag);
end
end

