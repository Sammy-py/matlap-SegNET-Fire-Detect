clc
clear 
load trained_attention_UNet
warning off
%v = VideoReader('video2.mp4');
v = VideoReader('video1.mp4');
k =1;
while hasFrame(v)
    a = imresize(readFrame(v),[256 256]);
    %k =k+1
    C = semanticseg(a,net);
    C1 = C=='fire';
%     sum(sum(double(C)))
    if sum(sum(double(C1)))>100
        disp('Fire')
    else
        disp('Nofire')
    end
     B = labeloverlay(a,C);
  imshow(B)
  drawnow
end
  
