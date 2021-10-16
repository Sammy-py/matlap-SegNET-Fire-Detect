
clc, clear

lgraph = layerGraph();

tempLayers = [
    imageInputLayer([256 256 3],"Name","ImageInputLayer")
    convolution2dLayer([3 3],64,"Name","Encoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-1-ReLU-1")
    convolution2dLayer([3 3],64,"Name","Encoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-1-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-1-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","Encoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-2-ReLU-1")
    convolution2dLayer([3 3],128,"Name","Encoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-2-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-2-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","Encoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-3-ReLU-1")
    convolution2dLayer([3 3],256,"Name","Encoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-3-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1,"Name","attconv1_2","Padding","same")
    batchNormalizationLayer("Name","attbatchnorm_1_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-3-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","Encoder-Stage-4-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-4-ReLU-1")
    convolution2dLayer([3 3],512,"Name","Encoder-Stage-4-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-4-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1,"Name","attconv1_1","Padding","same")
    batchNormalizationLayer("Name","attbatchnorm_1_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    dropoutLayer(0.5,"Name","Encoder-Stage-4-DropOut")
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-4-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],1024,"Name","Bridge-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Bridge-ReLU-1")
    convolution2dLayer([3 3],1024,"Name","Bridge-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Bridge-ReLU-2")
    dropoutLayer(0.5,"Name","Bridge-DropOut")
    transposedConv2dLayer([2 2],512,"Name","Decoder-Stage-1-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-UpReLU")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1,"Name","attconv2_1","Padding","same")
    batchNormalizationLayer("Name","attbatchnorm_2_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1,"Name","attconv1_4","Padding","same")
    batchNormalizationLayer("Name","attbatchnorm_1_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1")
    reluLayer("Name","relu_1")
    convolution2dLayer([1 1],1,"Name","attconv3_1","Padding","same")
    batchNormalizationLayer("Name","attbatchnorm_3_1")
    sigmoidLayer("sigmoid1_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = ElementWiseMultiplication(2, "elemMux_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-1-DepthConcatenation")
    convolution2dLayer([3 3],512,"Name","Decoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-ReLU-1")
    convolution2dLayer([3 3],512,"Name","Decoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-ReLU-2")
    transposedConv2dLayer([2 2],256,"Name","Decoder-Stage-2-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-UpReLU")];
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
    depthConcatenationLayer(2,"Name","Decoder-Stage-2-DepthConcatenation")
    convolution2dLayer([3 3],256,"Name","Decoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-ReLU-1")
    convolution2dLayer([3 3],256,"Name","Decoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-ReLU-2")
    transposedConv2dLayer([2 2],128,"Name","Decoder-Stage-3-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-UpReLU")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1,"Name","attconv2_3","Padding","same")
    batchNormalizationLayer("Name","attbatchnorm_2_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1,"Name","attconv1_3","Padding","same")
    batchNormalizationLayer("Name","attbatchnorm_1_3")];
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
    depthConcatenationLayer(2,"Name","Decoder-Stage-3-DepthConcatenation")
    convolution2dLayer([3 3],128,"Name","Decoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-ReLU-1")
    convolution2dLayer([3 3],128,"Name","Decoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-ReLU-2")
    transposedConv2dLayer([2 2],64,"Name","Decoder-Stage-4-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-4-UpReLU")];
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
    depthConcatenationLayer(2,"Name","Decoder-Stage-4-DepthConcatenation")
    convolution2dLayer([3 3],64,"Name","Decoder-Stage-4-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-4-ReLU-1")
    convolution2dLayer([3 3],64,"Name","Decoder-Stage-4-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-4-ReLU-2")
    convolution2dLayer([1 1],4,"Name","Final-ConvolutionLayer","Padding","same","WeightsInitializer","he")
    softmaxLayer("Name","Softmax-Layer")
    pixelClassificationLayer("Name","Segmentation-Layer")];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Encoder-Stage-1-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","attconv1_4");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Decoder-Stage-4-DepthConcatenation/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Encoder-Stage-2-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","attconv1_3");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Decoder-Stage-3-DepthConcatenation/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","attconv1_2");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","Encoder-Stage-3-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","Decoder-Stage-2-DepthConcatenation/in2");
lgraph = connectLayers(lgraph,"attbatchnorm_1_2","addition_2/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-4-ReLU-2","attconv1_1");
lgraph = connectLayers(lgraph,"Encoder-Stage-4-ReLU-2","Encoder-Stage-4-DropOut");
lgraph = connectLayers(lgraph,"Encoder-Stage-4-ReLU-2","Decoder-Stage-1-DepthConcatenation/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpReLU","attconv2_1");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpReLU","elemMux_1/in1");
lgraph = connectLayers(lgraph,"attbatchnorm_2_1","addition_1/in1");
lgraph = connectLayers(lgraph,"attbatchnorm_1_4","addition_4/in2");
lgraph = connectLayers(lgraph,"attbatchnorm_1_1","addition_1/in2");
lgraph = connectLayers(lgraph,"sigmoid1_1","elemMux_1/in2");
lgraph = connectLayers(lgraph,"elemMux_1","Decoder-Stage-1-DepthConcatenation/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpReLU","attconv2_2");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpReLU","elemMux_2/in1");
lgraph = connectLayers(lgraph,"attbatchnorm_2_2","addition_2/in1");
lgraph = connectLayers(lgraph,"sigmoid1_2","elemMux_2/in2");
lgraph = connectLayers(lgraph,"elemMux_2","Decoder-Stage-2-DepthConcatenation/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-UpReLU","attconv2_3");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-UpReLU","elemMux_3/in1");
lgraph = connectLayers(lgraph,"attbatchnorm_2_3","addition_3/in1");
lgraph = connectLayers(lgraph,"attbatchnorm_1_3","addition_3/in2");
lgraph = connectLayers(lgraph,"sigmoid1_3","elemMux_3/in2");
lgraph = connectLayers(lgraph,"elemMux_3","Decoder-Stage-3-DepthConcatenation/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-4-UpReLU","attconv2_4");
lgraph = connectLayers(lgraph,"Decoder-Stage-4-UpReLU","elemMux_4/in1");
lgraph = connectLayers(lgraph,"attbatchnorm_2_4","addition_4/in1");
lgraph = connectLayers(lgraph,"sigmoid1_4","elemMux_4/in2");
lgraph = connectLayers(lgraph,"elemMux_4","Decoder-Stage-4-DepthConcatenation/in1");

clear tempLayers;
% deepNetworkDesigner





