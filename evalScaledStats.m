startup
% load(fullfile(dir, 'results', 'nn45_40eps_lr_1e-1.mat'), 'nn45', 'nn45Info')
[PINE,trainIdx, testIdx] = loadPINE();
ndtStat = load(fullfile(dir, 'results', 'ndt_90eps_lr1e-1_bs10000_mmt1e-1_scaledStat.mat'));

L = PINE.data_all.X(:,5);
MLT = PINE.data_all.X(:,6);
Density = PINE.data_all.t;

LTrain = L(trainIdx);
MLTTrain = MLT(trainIdx);
DensityTrain = Density(trainIdx);
XTrain = PINE.data_all.X(trainIdx,:);
yTrain = Density(trainIdx);

XTest = PINE.data_all.X(testIdx,:);
LTest = L(testIdx);
MLTTest = MLT(testIdx);
DensityTest = PINE.data_all.t(testIdx);
yTest = PINE.data_all.t(testIdx);

Stat=ndtStat.Stat;

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
    if length(ind)>1
        cellMean = Stat(iCell).DensityMean;
        cellSTD = Stat(iCell).DensitySTD;
        yTest(ind) = (DensityTest(ind) - cellMean)/cellSTD;
    end
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

yNdt = predict(ndtStat.ndt, XTest);
yNdtStat = yNdt;
% reverse scale predicting output Density by cell mean and std
for iCell = 1:length(Stat)
    ind=find((LTest>=Stat(iCell).LRange(1))&(LTest<Stat(iCell).LRange(2))&...
            (MLTTest>=Stat(iCell).LonRange(1)/15)&(MLTTest<Stat(iCell).LonRange(2)/15));
    if length(ind)>1
        cellMean = Stat(iCell).DensityMean;
        cellSTD = Stat(iCell).DensitySTD;
        yNdtStat(ind) = yNdt(ind)*cellSTD + cellMean;
    end
end

rmseStat = sqrt(mean((yNdtStat - DensityTest).^2));

% Do not evaluate over Stat, evaluate the yPred with normalization and 
% plot the entire yPred (not Stat of yPred)
% StatNdt=getPlasmaSphereStats(LTest,MLTTest,DensityNdt);
% VisualizePlasmaSPhereStats(StatNdt, 'NDT')
% StatTest=getPlasmaSphereStats(LTest,MLTTest,DensityTest);
% VisualizePlasmaSPhereStats(StatTest, 'TestData')
visualizeDensity(DensityTest, yNdtStat, LTest, MLTTest, Stat, 'in StatScaled')


% comparison to non-Stat featured model:
ndtModel = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat'));
yNdtInGeneral = predict(ndtModel.ndt, XTest);
rmseInGeneral = sqrt(mean((yNdtInGeneral - DensityTest).^2));
visualizeDensity(DensityTest, yNdtInGeneral, LTest, MLTTest, Stat, 'in General')
