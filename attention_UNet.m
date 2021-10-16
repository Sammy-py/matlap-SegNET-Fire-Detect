clearvars; clc
%% edit these locations
imageDir = fullfile('resized_imgs/');
labelDir = fullfile('gt_imgs/');
imds = imageDatastore(imageDir);

classNames = ["background" "fire"];
labelIDs   = [0 1];% [0 255]

pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);
trainingData = pixelLabelImageSource(imds,pxds);
tbl = countEachLabel(trainingData);
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
classWeights = 1./frequency;

train_imds = pixelLabelImageDatastore(imds,pxds);



 lgraph = layerGraph();

    tempLayers = [
        imageInputLayer([256 256 3],"Name","inputImage")
        convolution2dLayer([3 3],64,"Name","encoder1_conv1","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","encoder1_bn_1")
        reluLayer("Name","encoder1_relu_1")
        convolution2dLayer([3 3],64,"Name","encoder1_conv2","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","encoder1_bn_2")
        reluLayer("Name","encoder1_relu_2")
        maxPooling2dLayer([2 2],"Name","encoder1_maxpool","HasUnpoolingOutputs",true,"Stride",[2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],1,"Name","attconv1_4","Padding","same")
        batchNormalizationLayer("Name","attbatchnorm_1_4")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],64,"Name","encoder2_conv1","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","encoder2_bn_1")
        reluLayer("Name","encoder2_relu_1")
        convolution2dLayer([3 3],64,"Name","encoder2_conv2","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","encoder2_bn_2")
        reluLayer("Name","encoder2_relu_2")
        maxPooling2dLayer([2 2],"Name","encoder2_maxpool","HasUnpoolingOutputs",true,"Stride",[2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],1,"Name","attconv1_3","Padding","same")
        batchNormalizationLayer("Name","attbatchnorm_1_3")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],64,"Name","encoder3_conv1","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","encoder3_bn_1")
        reluLayer("Name","encoder3_relu_1")
        convolution2dLayer([3 3],64,"Name","encoder3_conv2","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","encoder3_bn_2")
        reluLayer("Name","encoder3_relu_2")
        maxPooling2dLayer([2 2],"Name","encoder3_maxpool","HasUnpoolingOutputs",true,"Stride",[2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],64,"Name","encoder4_conv1","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","encoder4_bn_1")
        reluLayer("Name","encoder4_relu_1")
        convolution2dLayer([3 3],64,"Name","encoder4_conv2","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","encoder4_bn_2")
        reluLayer("Name","encoder4_relu_2")
        maxPooling2dLayer([2 2],"Name","encoder4_maxpool","HasUnpoolingOutputs",true,"Stride",[2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],1,"Name","attconv1_2","Padding","same")
        batchNormalizationLayer("Name","attbatchnorm_1_2")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],64,"Name","encoder5_conv1","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","encoder5_bn_1")
        reluLayer("Name","encoder5_relu_1")
        convolution2dLayer([3 3],64,"Name","encoder5_conv2","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","encoder5_bn_2")
        reluLayer("Name","encoder5_relu_2")
        maxPooling2dLayer([2 2],"Name","encoder5_maxpool","HasUnpoolingOutputs",true,"Stride",[2 2])
        maxUnpooling2dLayer("Name","decoder5_unpool")
        convolution2dLayer([3 3],64,"Name","decoder5_conv2","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","decoder5_bn_2")
        reluLayer("Name","decoder5_relu_2")
        convolution2dLayer([3 3],64,"Name","decoder5_conv1","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","decoder5_bn_1")
        reluLayer("Name","decoder5_relu_1")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],1,"Name","attconv1_1","Padding","same")
        batchNormalizationLayer("Name","attbatchnorm_1_1")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],1,"Name","attconv2_1","Padding","same")
        batchNormalizationLayer("Name","attbatchnorm_2_1")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","addition_1")
        reluLayer("Name","relu_1")
        convolution2dLayer([1 1],1,"Name","attconv3_1","Padding","same")
        batchNormalizationLayer("Name","attbatchnorm_3_1")
        sigmoidLayer("sigmoid1_1")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = ElementWiseMultiplication(2,"elemMux_1");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxUnpooling2dLayer("Name","decoder4_unpool")
        convolution2dLayer([3 3],64,"Name","decoder4_conv2","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","decoder4_bn_2")
        reluLayer("Name","decoder4_relu_2")
        convolution2dLayer([3 3],64,"Name","decoder4_conv1","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","decoder4_bn_1")
        reluLayer("Name","decoder4_relu_1")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],1,"Name","attconv2_2","Padding","same")
        batchNormalizationLayer("Name","attbatchnorm_2_2")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","addition_2")
        reluLayer("Name","relu_2")
        convolution2dLayer([1 1],1,"Name","attconv3_2","Padding","same")
        batchNormalizationLayer("Name","attbatchnorm_3_2")
        sigmoidLayer("sigmoid1_2")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = ElementWiseMultiplication(2, "elemMux_2");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxUnpooling2dLayer("Name","decoder3_unpool")
        convolution2dLayer([3 3],64,"Name","decoder3_conv2","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","decoder3_bn_2")
        reluLayer("Name","decoder3_relu_2")
        convolution2dLayer([3 3],64,"Name","decoder3_conv1","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","decoder3_bn_1")
        reluLayer("Name","decoder3_relu_1")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],1,"Name","attconv2_3","Padding","same")
        batchNormalizationLayer("Name","attbatchnorm_2_3")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","addition_3")
        reluLayer("Name","relu_3")
        convolution2dLayer([1 1],1,"Name","attconv3_3","Padding","same")
        batchNormalizationLayer("Name","attbatchnorm_3_3")
        sigmoidLayer("sigmoid1_3")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = ElementWiseMultiplication(2, "elemMux_3");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxUnpooling2dLayer("Name","decoder2_unpool")
        convolution2dLayer([3 3],64,"Name","decoder2_conv2","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","decoder2_bn_2")
        reluLayer("Name","decoder2_relu_2")
        convolution2dLayer([3 3],64,"Name","decoder2_conv1","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","decoder2_bn_1")
        reluLayer("Name","decoder2_relu_1")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],1,"Name","attconv2_4","Padding","same")
        batchNormalizationLayer("Name","attbatchnorm_2_4")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","addition_4")
        reluLayer("Name","relu_4")
        convolution2dLayer([1 1],1,"Name","attconv3_4","Padding","same")
        batchNormalizationLayer("Name","attbatchnorm_3_4")
        sigmoidLayer("sigmoid1_4")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = ElementWiseMultiplication(2, "elemMux_4");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxUnpooling2dLayer("Name","decoder1_unpool")
        convolution2dLayer([3 3],64,"Name","decoder1_conv2","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","decoder1_bn_2")
        reluLayer("Name","decoder1_relu_2")
        convolution2dLayer([3 3],numel(tbl.Name),"Name","decoder1_conv1","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
        batchNormalizationLayer("Name","decoder1_bn_1")
        reluLayer("Name","decoder1_relu_1")
        softmaxLayer("Name","softmax")];
    lgraph = addLayers(lgraph,tempLayers);


        pixelLabels = pixelClassificationLayer('Name','pixelLabels','ClassNames',tbl.Name,'ClassWeights',classWeights);

    
    lgraph = addLayers(lgraph,pixelLabels);

    lgraph = connectLayers(lgraph,"encoder1_maxpool/out","attconv1_4");
    lgraph = connectLayers(lgraph,"encoder1_maxpool/out","encoder2_conv1");
    lgraph = connectLayers(lgraph,"encoder1_maxpool/indices","decoder1_unpool/indices");
    lgraph = connectLayers(lgraph,"encoder1_maxpool/size","decoder1_unpool/size");
    lgraph = connectLayers(lgraph,"attbatchnorm_1_4","addition_4/in2");
    lgraph = connectLayers(lgraph,"encoder2_maxpool/out","attconv1_3");
    lgraph = connectLayers(lgraph,"encoder2_maxpool/out","encoder3_conv1");
    lgraph = connectLayers(lgraph,"encoder2_maxpool/indices","decoder2_unpool/indices");
    lgraph = connectLayers(lgraph,"encoder2_maxpool/size","decoder2_unpool/size");
    lgraph = connectLayers(lgraph,"attbatchnorm_1_3","addition_3/in2");
    lgraph = connectLayers(lgraph,"encoder3_maxpool/out","encoder4_conv1");
    lgraph = connectLayers(lgraph,"encoder3_maxpool/out","attconv1_2");
    lgraph = connectLayers(lgraph,"encoder3_maxpool/indices","decoder3_unpool/indices");
    lgraph = connectLayers(lgraph,"encoder3_maxpool/size","decoder3_unpool/size");
    lgraph = connectLayers(lgraph,"encoder4_maxpool/out","encoder5_conv1");
    lgraph = connectLayers(lgraph,"encoder4_maxpool/out","attconv1_1");
    lgraph = connectLayers(lgraph,"encoder4_maxpool/indices","decoder4_unpool/indices");
    lgraph = connectLayers(lgraph,"encoder4_maxpool/size","decoder4_unpool/size");
    lgraph = connectLayers(lgraph,"attbatchnorm_1_1","addition_1/in2");
    lgraph = connectLayers(lgraph,"encoder5_maxpool/indices","decoder5_unpool/indices");
    lgraph = connectLayers(lgraph,"encoder5_maxpool/size","decoder5_unpool/size");
    lgraph = connectLayers(lgraph,"decoder5_relu_1","attconv2_1");
    lgraph = connectLayers(lgraph,"decoder5_relu_1","elemMux_1/in2");
    lgraph = connectLayers(lgraph,"attbatchnorm_2_1","addition_1/in1");
    lgraph = connectLayers(lgraph,"sigmoid1_1","elemMux_1/in1");
    lgraph = connectLayers(lgraph,"elemMux_1","decoder4_unpool/in");
    lgraph = connectLayers(lgraph,"decoder4_relu_1","attconv2_2");
    lgraph = connectLayers(lgraph,"decoder4_relu_1","elemMux_2/in2");
    lgraph = connectLayers(lgraph,"attbatchnorm_2_2","addition_2/in1");
    lgraph = connectLayers(lgraph,"attbatchnorm_1_2","addition_2/in2");
    lgraph = connectLayers(lgraph,"sigmoid1_2","elemMux_2/in1");
    lgraph = connectLayers(lgraph,"elemMux_2","decoder3_unpool/in");
    lgraph = connectLayers(lgraph,"decoder3_relu_1","attconv2_3");
    lgraph = connectLayers(lgraph,"decoder3_relu_1","elemMux_3/in2");
    lgraph = connectLayers(lgraph,"attbatchnorm_2_3","addition_3/in1");
    lgraph = connectLayers(lgraph,"sigmoid1_3","elemMux_3/in1");
    lgraph = connectLayers(lgraph,"elemMux_3","decoder2_unpool/in");
    lgraph = connectLayers(lgraph,"decoder2_relu_1","attconv2_4");
    lgraph = connectLayers(lgraph,"decoder2_relu_1","elemMux_4/in2");
    lgraph = connectLayers(lgraph,"attbatchnorm_2_4","addition_4/in1");
    lgraph = connectLayers(lgraph,"sigmoid1_4","elemMux_4/in1");
    lgraph = connectLayers(lgraph,"elemMux_4","decoder1_unpool/in");
    lgraph = connectLayers(lgraph,"softmax","pixelLabels");

    clear tempLayers;

options = trainingOptions('sgdm', 'InitialLearnRate', 0.1,'MaxEpochs', 50,'MiniBatchSize',32,'VerboseFrequency', 1,...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',0.05,'LearnRateDropPeriod',10, 'shuffle', 'every-epoch',...
    'ExecutionEnvironment','multi-gpu');

tStart = tic;
net = trainNetwork(train_imds, lgraph, options);
tEnd = toc(tStart);
disp(tEnd)

save ('trained_attention_UNet.mat', 'net')

%% save obtained results to a folder

mkdir results_A_UNet

close all
for i=1
    a=imread(imds.Files{i}); 
    C = semanticseg(a,net);
%     bw=im2uint8(C == classNames(2));
%     fname = strcat('results_A_UNet/',imds.Files{i}(1,end-10:end-4),'.png');
%     imwrite(bw,fname);
end
