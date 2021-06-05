startup

[PINE,trainIdx, testIdx] = loadPINE();
MLT = PINE.data_all.X(:,6); polarMLT = zeros(size(MLT,1),2);
polarMLT(:,1) = cos(MLT*15*pi/180);
polarMLT(:,2) = sin(MLT*15*pi/180);
X = [PINE.data_all.X(:,1:5), polarMLT, PINE.data_all.X(:,7:end)];
L = PINE.data_all.X(:,5);
Density = PINE.data_all.t;

LTrain = L(trainIdx);
MLTTrain = MLT(trainIdx);
DensityTrain = Density(trainIdx);
XTrain = X(trainIdx,:);
yTrain = Density(trainIdx);

XTest = X(testIdx,:);
LTest = L(testIdx);
MLTTest = MLT(testIdx);
DensityTest = Density(testIdx);
yTest = Density(testIdx);

Stat=getPlasmaSphereStatsL(LTrain,yTrain);
% VisualizePlasmaSPhereStats(Stat)

% --------------------- weighted ---------------------
% % weighted training output Density by inverse of nDataPoint
% dropIdx = [];
% for iCell = 1:length(Stat)
%     ind=find((LTrain>=Stat(iCell).LRange(1))&(LTrain<Stat(iCell).LRange(2)));
%     cellnDataPoint = Stat(iCell).nDataPoint/300;   
%     yTrain(ind) = DensityTrain(ind)/sqrt(cellnDataPoint);
% end
% undersampleIdx = setdiff((1:length(yTrain))', dropIdx);
% LTrain = LTrain(undersampleIdx);
% MLTTrain = MLTTrain(undersampleIdx);
% XTrain = XTrain(undersampleIdx, :);
% yTrain = yTrain(undersampleIdx);
% 
% % for debug:
% % StatAfter=getPlasmaSphereStatsL(LTrain,yTrain);
% % VisualizePlasmaSPhereStats(StatScaled)
% 
% % weight testing output Density by inverse of nDataPoint
% for iCell = 1:length(Stat)
%     ind=find((LTest>=Stat(iCell).LRange(1))&(LTest<Stat(iCell).LRange(2)));
%     cellnDataPoint = Stat(iCell).nDataPoint/300;   
%     yTest(ind) = DensityTrain(ind)/sqrt(cellnDataPoint);
% end
% 
% StatWeighted=getPlasmaSphereStatsL(LTrain,yTrain);
% 
% % for debug
% StatWeightedTest=getPlasmaSphereStatsL(LTest,yTest);
% % VisualizePlasmaSPhereStats(Stat)

% ------------------- Stat standardized then weighted -------------------
% scale training output Density by cell mean and std
dropIdx = [];
for iCell = 1:length(Stat)
    ind=find((LTrain>=Stat(iCell).LRange(1))&(LTrain<Stat(iCell).LRange(2)));
    cellMean = Stat(iCell).DensityMean;
    cellSTD = Stat(iCell).DensitySTD;
    cellWeight = sqrt(Stat(iCell).nDataPoint)/100;
    if length(ind) <=1
        dropIdx = union(dropIdx, ind);
    else
        yTrain(ind) = (DensityTrain(ind) - cellMean)/(cellSTD*cellWeight);
    end
end
undersampleIdx = setdiff((1:length(yTrain))', dropIdx);
LTrain = LTrain(undersampleIdx);
MLTTrain = MLTTrain(undersampleIdx);
XTrain = XTrain(undersampleIdx, :);
yTrain = yTrain(undersampleIdx);

% scale testing output Density by cell mean and std
for iCell = 1:length(Stat)
    ind=find((LTest>=Stat(iCell).LRange(1))&(LTest<Stat(iCell).LRange(2)));
    cellMean = Stat(iCell).DensityMean;
    cellSTD = Stat(iCell).DensitySTD;
    cellWeight = sqrt(Stat(iCell).nDataPoint)/100;
    yTest(ind) = (DensityTest(ind) - cellMean)/(cellSTD*cellWeight);
end

% for debug:
StatWeightedStandarized=getPlasmaSphereStatsL(LTrain,yTrain);
StatWeightedStandarizedTest=getPlasmaSphereStatsL(LTest,yTest);
% VisualizePlasmaSPhereStats(StatWeightedStandarized)


% --------------------- minmax and remove const ---------------------
%pre- and post- processing
procFcnsInput = {}; procFcnsOutput = {};
procFcnsInput{1} = 'removeconstantrows'; 
procFcnsInput{2} = 'mapminmax';

[XTrain, settingsXTrain] = preProcess(XTrain, procFcnsInput);
XTest = preProcessApply(XTest, procFcnsInput, settingsXTrain);

% ind = find(isnan(yTrain));
% XTrain = XTrain(setdiff(1:length(yTrain),ind), :);
% yTrain = yTrain(setdiff(1:length(yTrain),ind));

% --------------------- Call Model ---------------------

% call ndtSGD:
% ndtSGD
% 
% names = sprintf('ndt15_50eps_lr1e-1_bs10000_mmt1e-1_scaledStatLPolarMLT.mat');
% 
% save(fullfile(dir, 'results', names), 'ndt', 'ndtInfo',...
%     'procFcnsInput', 'settingsXTrain', 'Stat')
% names = sprintf('ndt64Adam_50eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1_scaledStatLpolarMLT.mat');
% save(fullfile(dir, 'results','testModel', names), 'ndt', 'ndtInfo',...
%      'procFcnsInput', 'settingsXTrain', 'Stat')
%  
% call nn45SGD:
% nn45SGD
% 
% names = sprintf('nn45_200eps_lr1e-1_bs10000_mmt1e-1_scaledStatLPolarMLT.mat');
% 
% save(fullfile(dir, 'results', names), 'nn45', 'nn45Info',...
%     'procFcnsInput', 'settingsXTrain', 'Stat')

% call ndtLM:
ndtLM
names = sprintf('ndt_40eps_statLscaledWeightedPolarMLT.mat');
save(fullfile(dir, 'results', 'trainlm', names), 'ndt', 'ndtInfo',...
    'procFcnsInput', 'settingsXTrain', 'Stat')

% call nn45LM
% nn45LM
% 
% names = sprintf('nn45_40eps_scaledStatLPolarMLT.mat');
% save(fullfile(dir, 'results','trainlm', names), 'nn45', 'nn45Info',...
%     'procFcnsInput', 'settingsXTrain', 'Stat')



