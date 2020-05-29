%本程序用于对数据进行去噪和归一化处理
%读入数据
filename='D:\Preprocessing\data_25var.csv';
Data_raw=csvread(filename,1,0,[1,0,27000,24]);
Data_mean=mean(Data_raw,1);
Data_std=std(Data_raw,0,1);
Data_cut=Data_raw(:,:);
for j=1:25
    for i=1:27000
        if Data_raw(i,j)>Data_mean(:,j)+3*Data_std(:,j)||Data_raw(i,j)<Data_mean(:,j)-3*Data_std(:,j)
            Data_cut(i,j)=Data_mean(:,j);
        end
    end
end
Data_ma=Data_cut(:,:);
for j=1:25
    for i=5:27000
        Data_ma(i,j)=mean(Data_cut(i-4:i,j),1);
    end
end
for i=1:25
    plot(Data_raw(5500:13500,i),'-g')
    hold on
    plot(Data_cut(5500:13500,i),'-b')
    hold on
    plot(Data_ma(5500:13500,i),'-r')
    legend('Original','3sigma','3sigma+MA');
    hold off
    filename=num2str(i);
    saveas(gcf,filename,'jpg')
end
%Normalize the data
Max_=max(Data_ma);
Min_=min(Data_ma);
for i=1:25
    Data_norm(1:27000,i)=(Data_ma(1:27000,i)-Min_(i))/(Max_(i)-Min_(i));
end
xlswrite('Data_8000_cut_ma.xlsx',Data_ma(5500:13500,:),1,'A1')
xlswrite('Data_8000_normalized.xlsx',Data_norm(5500:13500,:),1,'A1')
meanresult=mean(Data_ma(5500:13500,:),1);
