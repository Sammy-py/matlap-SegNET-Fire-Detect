dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
imageDir = fullfile(dataSetDir,'trainingImages');
labelDir = fullfile(dataSetDir,'trainingLabels');

imds = imageDatastore(imageDir);

classNames = ["background" "case1" "case2" "case3"];
labelIDs   = [0 85 170 255];

pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

imageSize = [512 512];
numClasses = 4;
lgraph = unetLayers(imageSize, numClasses)

train_imds = pixelLabelImageDatastore(imds,pxds);