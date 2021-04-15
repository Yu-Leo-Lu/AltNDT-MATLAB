startup

[PINE,trainIdx, testIdx] = loadPINE();
ndtStatPolar = load(fullfile(dir, 'results', 'ndt_191eps_lr1e-1_bs10000_mmt1e-1_scaledStatPolarMLT'));
nn45StatPolar = load(fullfile(dir, 'results', 'nn45_200eps_lr1e-1_bs10000_mmt1e-1_scaledStatPolarMLT'));

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

Stat=ndtStatPolar.Stat;
% VisualizePlasmaSPhereStats(Stat)

XTestRaw = X(testIdx,:);
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
% compare with polarMLT only
ndtPolar = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_polarMLT.mat'));
nn45Polar = load(fullfile(dir, 'results', 'nn45_200eps_lr1e-1_bs10000_mmt1e-1_polarMLT.mat'));

% compare with non-StatPolar
ndtModel = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat'));
nn45Model = load(fullfile(dir, 'results', 'nn45_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat'));

inputLabelsGnl = PINE.feature_names;
XTrainGnl = PINE.data_all.X(trainIdx,:);
yTrainGnl = PINE.data_all.t(trainIdx);
[XTrainGnl, settingsXTrainGnl] = preProcess(XTrainGnl, procFcnsInput);
XTestGnlRaw = PINE.data_all.X(testIdx,:);

% prediction in events from 2001
load(fullfile(dir, 'PINE_data', 'data2001'));

DateStr = '27-Jun-2001'; hrs = 11;
days = datenum(DateStr) - datenum(2001,1,1);
xTryGnl = data2001(days*24+hrs,:);
xTry = [xTryGnl(1:6),nan,xTryGnl(7:end)];

figure; % by one testing data
subplot(2,3,2);
VisualizePredictionPlasmaSphere(ndtPolar.ndt,inputLabels,xTry,12,36,procFcnsInput,settingsXTrain,[], 'in ndtPolar')
subplot(2,3,5);
VisualizePredictionPlasmaSphere(nn45Polar.nn45,inputLabels,xTry,12,36,procFcnsInput,settingsXTrain,[], 'in nn45Polar')
subplot(2,3,1);
VisualizePredictionPlasmaSphere(ndtModel.ndt,inputLabelsGnl,xTryGnl,12,36,procFcnsInput,settingsXTrainGnl,[],'in General ndt')
subplot(2,3,4);
VisualizePredictionPlasmaSphere(nn45Model.nn45,inputLabelsGnl,xTryGnl,12,36,procFcnsInput,settingsXTrainGnl,[],'in General nn45')
subplot(2,3,3);
VisualizePredictionPlasmaSphere(ndtStatPolar.ndt,inputLabels,xTry,12,36,procFcnsInput,settingsXTrain,Stat, 'in ndtStatPolar')
subplot(2,3,6);
VisualizePredictionPlasmaSphere(nn45StatPolar.nn45,inputLabels,xTry,12,36,procFcnsInput,settingsXTrain,Stat, 'in nn45StatPolar')
sgtitle([DateStr,' ',num2str(hrs),':00']);

% find idx by kp, X(:,2)
% idx = find(XTestRaw(:,2)==4.3);
% half open shape
% {266016, 266528,266540,(265985 XL ae)} kp 8.3
% nn45 266540 graph is inconsistent between general, polar and statpolar 

% open wider kp 4.3
% 184064,314289,323827 

% almost all open kp 3.3
% 72284 237483
% iTry = idx(randi(length(idx),1)); %150000 % %194945 %184064 open wider
% iTry = 266540;

% find index by most variant pc:
% !!!!!!!!!!!!!!!!!!!!!
% need to FIX!
% PCA does not include L, MLT
% predict using STAT, when no stat? use overall mean/std


% standarlized and PCA
XTestNormalized = [XTestRaw(:, 1:4), XTestRaw(:, 8:end)];
XTestMean = nanmean(XTestNormalized); XTestStd = nanstd(XTestNormalized);
XTestNormalized = XTestNormalized - XTestMean;
XTestNormalized = XTestNormalized./XTestStd;
[coeff,score,latent,~,explained] = pca(XTestNormalized);

% select principal component (PC) and variation option 'max' or 'min'
PC = 1;
varOption = 'max';
fh = str2func(varOption);
idx = find(score(:,PC) == fh(score(:,PC)));
iTry = idx(randi(length(idx),1)); 
xTry = score(iTry,:);
xTry(:,setdiff(1:28, PC)) = 0;
xTry = xTry*coeff';
xTry = xTry.*XTestStd+XTestMean;
xTry = [xTry(1:4),1,1,1, xTry(5:end)];
xTryGnl = [xTry(1:4),1,1,xTry(8:end)];

figure; % by one testing data
subplot(2,3,2);
VisualizePredictionPlasmaSphere(ndtPolar.ndt,inputLabels,xTry,12,36,procFcnsInput,settingsXTrain,[], 'in ndtPolar')
subplot(2,3,5);
VisualizePredictionPlasmaSphere(nn45Polar.nn45,inputLabels,xTry,12,36,procFcnsInput,settingsXTrain,[], 'in nn45Polar')
subplot(2,3,1);
VisualizePredictionPlasmaSphere(ndtModel.ndt,inputLabelsGnl,xTryGnl,12,36,procFcnsInput,settingsXTrainGnl,[],'in General ndt')
subplot(2,3,4);
VisualizePredictionPlasmaSphere(nn45Model.nn45,inputLabelsGnl,xTryGnl,12,36,procFcnsInput,settingsXTrainGnl,[],'in General nn45')
subplot(2,3,3);
VisualizePredictionPlasmaSphere(ndtStatPolar.ndt,inputLabels,xTry,12,36,procFcnsInput,settingsXTrain,Stat, 'in ndtStatPolar')
subplot(2,3,6);
VisualizePredictionPlasmaSphere(nn45StatPolar.nn45,inputLabels,xTry,12,36,procFcnsInput,settingsXTrain,Stat, 'in nn45StatPolar')
sgtitle([varOption,' Principal Component ', num2str(PC)]);

% figure; % by indices
% subplot(2,3,2);
% VisualizePredictionPlasmaSphere(ndtPolar.ndt,inputLabels,XTestRaw(iTry,:),12,36,procFcnsInput,settingsXTrain,[], 'in ndtPolar')
% subplot(2,3,5);
% VisualizePredictionPlasmaSphere(nn45Polar.nn45,inputLabels,XTestRaw(iTry,:),12,36,procFcnsInput,settingsXTrain,[], 'in nn45Polar')
% subplot(2,3,1);
% VisualizePredictionPlasmaSphere(ndtModel.ndt,inputLabelsGnl,XTestGnlRaw(iTry,:),12,36,procFcnsInput,settingsXTrainGnl,[],'in General ndt')
% subplot(2,3,4);
% VisualizePredictionPlasmaSphere(nn45Model.nn45,inputLabelsGnl,XTestGnlRaw(iTry,:),12,36,procFcnsInput,settingsXTrainGnl,[],'in General nn45')
% subplot(2,3,3);
% VisualizePredictionPlasmaSphere(ndtStatPolar.ndt,inputLabels,XTestRaw(iTry,:),12,36,procFcnsInput,settingsXTrain,Stat, 'in ndtStatPolar')
% subplot(2,3,6);
% VisualizePredictionPlasmaSphere(nn45StatPolar.nn45,inputLabels,XTestRaw(iTry,:),12,36,procFcnsInput,settingsXTrain,Stat, 'in nn45StatPolar')
% sgtitle('max PC1');


% Reconstructed Data:
% nPc = 5;
% XTestPC = ((XTestRaw)*coeff(:,1:nPc)*(coeff(:,1:nPc)'));
% XTestGnlPC = ((XTestGnlRaw)*coeffGnl(:,1:nPc)*(coeffGnl(:,1:nPc)'));

% subplot(2,3,2);
% VisualizePredictionPlasmaSphere(ndtPolar.ndt,inputLabels,XTestPC(iTry,:),12,36,procFcnsInput,settingsXTrain,[], 'in ndtPolar')
% subplot(2,3,5);
% VisualizePredictionPlasmaSphere(nn45Polar.nn45,inputLabels,XTestPC(iTry,:),12,36,procFcnsInput,settingsXTrain,[], 'in nn45Polar')
% 
% subplot(2,3,1);
% VisualizePredictionPlasmaSphere(ndtModel.ndt,inputLabelsGnl,XTestGnlPC(iTry,:),12,36,procFcnsInput,settingsXTrainGnl,[],'in General ndt')
% subplot(2,3,4);
% VisualizePredictionPlasmaSphere(nn45Model.nn45,inputLabelsGnl,XTestGnlPC(iTry,:),12,36,procFcnsInput,settingsXTrainGnl,[],'in General nn45')
% 
% subplot(2,3,3);
% VisualizePredictionPlasmaSphere(ndtStatPolar.ndt,inputLabels,XTestPC(iTry,:),12,36,procFcnsInput,settingsXTrain,Stat, 'in ndtStatPolar')
% subplot(2,3,6);
% VisualizePredictionPlasmaSphere(nn45StatPolar.nn45,inputLabels,XTestPC(iTry,:),12,36,procFcnsInput,settingsXTrain,Stat, 'in nn45StatPolar')



% for debugging in one testing obv:
% sqrt(mean((predict(nn45Model.nn45, XTrainNS(1:10000, :))-yTrainNS(1:10000)).^2));
% xTry = XTestNSRaw(1,:);
% xTry(5) = 3; xTry(6) = 359/15;
% xTry = preProcessApply(xTry, procFcnsInput, settingsXTrainNS);
% disp(['nn45: ', num2str(predict(nn45Model.nn45, xTry))])
% disp(['ndt: ', num2str(predict(ndtModel.ndt, xTry))])
% max(PINE.data_all.X(:,6)); % problem found, MLT is time 0-24, not degree
