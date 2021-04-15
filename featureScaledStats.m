startup

[PINE,trainIdx, testIdx] = loadPINE();
L = PINE.data_all.X(:,5);
MLT = PINE.data_all.X(:,6);
Density = PINE.data_all.t;

LTrain = L(trainIdx);
MLTTrain = MLT(trainIdx);
DensityTrain = Density(trainIdx);
XTrain = PINE.data_all.X(trainIdx,:);
yTrain = Density(trainIdx);

% undersample in training, drop data with 5<=L<=7
% dropIdx = find((LTrain>=5.5)&(LTrain<=6));
% dropIdx = randsample(dropIdx,ceil(length(dropIdx)/2));
% undersampleIdx = setdiff((1:length(yTrain))', dropIdx);
% 
% LTrain = LTrain(undersampleIdx);
% MLTTrain = MLTTrain(undersampleIdx);
% XTrain = XTrain(undersampleIdx, :);
% yTrain = yTrain(undersampleIdx);

% another drop
% dropIdx = find((LTrain>=5)&(LTrain<=5.5));
% dropIdx = randsample(dropIdx,ceil(length(dropIdx)/4));
% undersampleIdx = setdiff((1:length(yTrain))', dropIdx);
% 
% LTrain = LTrain(undersampleIdx);
% MLTTrain = MLTTrain(undersampleIdx);
% XTrain = XTrain(undersampleIdx, :);
% yTrain = yTrain(undersampleIdx);

Stat=getPlasmaSphereStats(LTrain,MLTTrain,yTrain);
% VisualizePlasmaSPhereStats(Stat)

XTest = PINE.data_all.X(testIdx,:);
LTest = L(testIdx);
MLTTest = MLT(testIdx);
DensityTest = PINE.data_all.t(testIdx);
yTest = PINE.data_all.t(testIdx);

% scale training output Density by cell mean and std
dropIdx = [];
for iCell = 1:length(Stat)
    ind=find((LTrain>=Stat(iCell).LRange(1))&(LTrain<Stat(iCell).LRange(2))&...
            (MLTTrain>=Stat(iCell).LonRange(1)/15)&(MLTTrain<Stat(iCell).LonRange(2)/15));
    cellMean = Stat(iCell).DensityMean;
    cellSTD = Stat(iCell).DensitySTD;
    if length(ind) <=1
        dropIdx = union(dropIdx, ind);
    else
        yTrain(ind) = (DensityTrain(ind) - cellMean)/cellSTD;
    end
end
undersampleIdx = setdiff((1:length(yTrain))', dropIdx);
LTrain = LTrain(undersampleIdx);
MLTTrain = MLTTrain(undersampleIdx);
XTrain = XTrain(undersampleIdx, :);
yTrain = yTrain(undersampleIdx);

% scale testing output Density by cell mean and std
for iCell = 1:length(Stat)
    ind=find((LTest>=Stat(iCell).LRange(1))&(LTest<Stat(iCell).LRange(2))&...
            (MLTTest>=Stat(iCell).LonRange(1)/15)&(MLTTest<Stat(iCell).LonRange(2)/15));
    cellMean = Stat(iCell).DensityMean;
    cellSTD = Stat(iCell).DensitySTD;
    yTest(ind) = (DensityTest(ind) - cellMean)/cellSTD;
end

StatScaled=getPlasmaSphereStats(LTrain,MLTTrain,yTrain);
VisualizePlasmaSPhereStats(StatScaled)

%pre- and post- processing
procFcnsInput = {}; procFcnsOutput = {};
procFcnsInput{1} = 'removeconstantrows'; 
procFcnsInput{2} = 'mapminmax';
% procFcnsOutput{1} = 'removeconstantrows'; 
% procFcnsOutput{2} = 'mapminmax';

[XTrain, settingsXTrain] = preProcess(XTrain, procFcnsInput);
% [yTrain, settingstTrain] = preProcess(yTrain, procFcnsOutput);
XTest = preProcessApply(XTest, procFcnsInput, settingsXTrain);
% yTest = preProcessApply(yTest, procFcnsOutput, settingstTrain);

% ind = find(isnan(yTrain));
% XTrain = XTrain(setdiff(1:length(yTrain),ind), :);
% yTrain = yTrain(setdiff(1:length(yTrain),ind));

% call NDT:
% ndtSGD

names = sprintf('ndt_200eps_lr1e-1_bs10000_mmt1e-1_statFeat.mat');

save(fullfile(dir, 'results', names), 'ndt', 'ndtInfo',...
    'procFcnsInput', 'settingsXTrain', 'Stat')

% call NN45:
% nn45SGD







