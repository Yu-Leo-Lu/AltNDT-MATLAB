startup

[PINE,trainIdx, testIdx] = loadPINE();
ndtStatPolar = load(fullfile(dir, 'results', 'ndt_191eps_lr1e-1_bs10000_mmt1e-1_scaledStatPolarMLT'));
MLT = PINE.data_all.X(:,6); polarMLT = zeros(size(MLT,1),2);
polarMLT(:,1) = cos(MLT*15*pi/180);
polarMLT(:,2) = sin(MLT*15*pi/180);
X = [PINE.data_all.X(:,1:5), polarMLT, PINE.data_all.X(:,7:end)];
L = PINE.data_all.X(:,5);
Density = PINE.data_all.t;
inputLabels = [PINE.feature_names(1:5); 'cmlt'; 'smlt'; PINE.feature_names(7:end)];

LTrain = L(trainIdx);
MLTTrain = MLT(trainIdx);
DensityTrain = Density(trainIdx);
XTrain = X(trainIdx,:);
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

Stat=ndtStatPolar.Stat;
% VisualizePlasmaSPhereStats(Stat)

XTest = X(testIdx,:);
LTest = L(testIdx);
MLTTest = MLT(testIdx);
DensityTest = Density(testIdx);
yTest = Density(testIdx);

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

% StatScaled=getPlasmaSphereStats(LTrain,MLTTrain,yTrain);
% VisualizePlasmaSPhereStats(StatScaled)

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

yNdt = predict(ndtStatPolar.ndt, XTest);
yNdtStatPolar = yNdt;
% reverse scale predicting output Density by cell mean and std
for iCell = 1:length(Stat)
    ind=find((LTest>=Stat(iCell).LRange(1))&(LTest<Stat(iCell).LRange(2))&...
            (MLTTest>=Stat(iCell).LonRange(1)/15)&(MLTTest<Stat(iCell).LonRange(2)/15));
    if length(ind)>1
        cellMean = Stat(iCell).DensityMean;
        cellSTD = Stat(iCell).DensitySTD;
        yNdtStatPolar(ind) = yNdt(ind)*cellSTD + cellMean;
    end
end
visualizeDensity(DensityTest, yNdtStatPolar, LTest, MLTTest, Stat, 'in Scaled Stat and Polar MLT')


yNN45 = predict(nn45StatPolar.nn45, XTest);
yNN45StatPolar = yNN45;
% reverse scale predicting output Density by cell mean and std
for iCell = 1:length(Stat)
    ind=find((LTest>=Stat(iCell).LRange(1))&(LTest<Stat(iCell).LRange(2))&...
            (MLTTest>=Stat(iCell).LonRange(1)/15)&(MLTTest<Stat(iCell).LonRange(2)/15));
    if length(ind)>1
        cellMean = Stat(iCell).DensityMean;
        cellSTD = Stat(iCell).DensitySTD;
        yNN45StatPolar(ind) = yNN45(ind)*cellSTD + cellMean;
    end
end
visualizeDensity(DensityTest, yNN45StatPolar, LTest, MLTTest, Stat, 'in Scaled Stat and Polar MLT (nn45)')

