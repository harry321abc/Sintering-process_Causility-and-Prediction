%本程序用于进行核密度估计
%读入数据
filename='D:\本科毕设\Preprocessing\Data_8000_cut_ma.csv';
data=csvread(filename,0,19,[0,19,8000,24]);
x=data(:,6);
[f, xi] = ksdensity(x);
F=zeros(1,100);
for i=2:100
    F(1,i)=trapz(xi(1:i),f(1:i));
end
subplot(1,2,1)
plot(xi,f,'--*r')
subplot(1,2,2)
plot(xi,F,':o')