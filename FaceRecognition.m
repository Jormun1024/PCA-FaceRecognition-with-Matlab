clc;
clear;
close all;

eF = 20;%特征向量数，一定程度上反应识别精度
faceW = 64;
faceH = 64;
numPerLine = 11; 
ShowLine = 15;

%加载数据集
load('Yale_64x64.mat');
F = zeros(faceH*ShowLine,faceW*numPerLine); 
for i=0:ShowLine-1
    for j=0:numPerLine-1
        F(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(fea(i*numPerLine+j+1,:),[faceH,faceW]); 
    end
end
figure(1)
imagesc(F);
colormap(gray);

%选择进行测试的人脸坐标
true = 0;
while true==0
    a=input("请选择进行测试的横人脸坐标：\n");
    if a>0 && a<12
        true = 1;
    else
        disp("错误Error");
    end 
end
a=a-1;
true = 0;
while true==0
    b=input("请选择进行测试的纵人脸坐标：\n");
    if b>0 && b<16
        true = 1;
    else
        disp("错误Error");
    end 
end
b=b-1;
f = F(b*faceH+1:(b+1)*faceH,a*faceW+1:(a+1)*faceW);
figure(2)
imagesc(f);
colormap(gray);

%剔除测试数据，整理数据集
F(b*faceH+1:(b+1)*faceH,a*faceW+1:(a+1)*faceW) = 255-f;
for i=0:ShowLine-1
    for j=0:numPerLine-1
        traindata(i*numPerLine+j+1,:) = reshape(F(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW),[],1); 
    end
end
figure(3)
imagesc(F);
colormap(gray);

%实现PCA降维
%矩阵去中心化
AVG = zscore(traindata')';
fav = zscore(f')';

%计算协方差
COV = (1/165)*AVG*(AVG');
cov = (1/64)*fav*(fav');

%求协方差矩阵的特征值与特征向量
[V,D] = eig(COV);
EE = abs(diag(D));

%取训练集的前eF个特征向量构成特征脸
for i=1:eF
    VG(:,i)=V(:,i);
end
EV=AVG'*VG;
for j=1:eF
    tempA=reshape(EV(:,j),64, []);
    if(j==1)
        EVI=tempA;
    else
        EVI=[EVI,tempA];%将特征脸拼接成一个图片
    end
end

%计算测试图像在特征区域的投影坐标
testdate = reshape(fav,1,[]);
W=(testdate)*EV;

%在特征区域寻找最小欧氏距离
distance=zeros(165,1);
for i=1:165
    tempW=VG(i,:);
    distance(i)=pdist2(W,tempW);
end
[m,index]=min(distance);
resultimg=reshape(traindata(index,:),[faceH,faceW]);
figure(4)
subplot(1,2,1)
imshow(f,[]);
title('被识别的人像')
subplot(1,2,2)
imshow(resultimg,[]);
title('匹配到的人像')