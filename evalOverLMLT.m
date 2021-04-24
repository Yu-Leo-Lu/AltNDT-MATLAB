startup

[PINE,trainIdx, testIdx] = loadPINE();
inputLabelsGnl = PINE.feature_names;
inputLabels = [PINE.feature_names(1:5); 'cmlt'; 'smlt'; PINE.feature_names(7:end)];

% load models:
% models
ndtStatLPolar = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_scaledStatLPolarMLT'));
nn45StatLPolar = load(fullfile(dir, 'results', 'nn45_200eps_lr1e-1_bs10000_mmt1e-1_scaledStatLPolarMLT'));

% with polarMLT only
ndtPolar = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_polarMLT.mat'));
nn45Polar = load(fullfile(dir, 'results', 'nn45_200eps_lr1e-1_bs10000_mmt1e-1_polarMLT.mat'));

% with non-StatPolar
ndtModel = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat'));
nn45Model = load(fullfile(dir, 'results', 'nn45_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat'));

% LM models
nn45LM = load(fullfile(dir,'results','trainlm','nn45_40eps.mat'));
ndtLM = load(fullfile(dir,'results','trainlm','ndt_40eps.mat'));
ndtPolarLM = load(fullfile(dir, 'results','trainlm', 'ndt_40eps_polarMLT.mat'));
ndtStatLPolarLM = load(fullfile(dir, 'results','trainlm', 'ndt_40eps_scaledStatLPolarMLT.mat'));


% prediction in events from 2001
load(fullfile(dir, 'PINE_data', 'data2001'));
load(fullfile(dir, 'PINE_data', 'data2001Hourly'));

DateStr = '10-Jun-2001'; hrs = 06; mins = 33;
days = datenum(DateStr) - datenum(2001,1,1);
xTryGnl = data2001(days*24*60+hrs*60+mins,:);
xTry = [xTryGnl(1:6),nan,xTryGnl(7:end)];

% xTryGnlH = data2001Hourly(days*24+hrs,:);
% xTryH = [xTryGnlH(1:6),nan,xTryGnlH(7:end)];
% xTryGstack = [xTryGnl;xTryGnlH];

figure; % by one testing data xTry or xTryGnl
subplot(2,3,1);
VisualizePredictionPlasmaSphere(ndtModel.ndt,inputLabelsGnl,xTryGnl,12,72,ndtModel.procFcnsInput,ndtModel.settingsXTrain,[],'in General ndt',1.3)
subplot(2,3,2);
VisualizePredictionPlasmaSphere(ndtPolar.ndt,inputLabels,xTry,12,72,ndtPolar.procFcnsInput,ndtPolar.settingsXTrain,[], 'in Polar ndt')
subplot(2,3,3);
VisualizePredictionPlasmaSphere(ndtStatLPolar.ndt,inputLabels,xTry,12,72,ndtStatLPolar.procFcnsInput,ndtStatLPolar.settingsXTrain,ndtStatLPolar.Stat, 'in Stat scaled Polar ndt')
subplot(2,3,4);
VisualizePredictionPlasmaSphere(nn45Model.nn45,inputLabelsGnl,xTryGnl,12,72,nn45Model.procFcnsInput,nn45Model.settingsXTrain,[],'in General nn45')
subplot(2,3,5);
VisualizePredictionPlasmaSphere(nn45Polar.nn45,inputLabels,xTry,12,72,nn45Polar.procFcnsInput,nn45Polar.settingsXTrain,[], 'in Polar nn45')
subplot(2,3,6);
VisualizePredictionPlasmaSphere(nn45StatLPolar.nn45,inputLabels,xTry,12,72,nn45StatLPolar.procFcnsInput,nn45StatLPolar.settingsXTrain,nn45StatLPolar.Stat, 'in Stat scaled Polar nn45')
sgtitle([DateStr,' ',num2str(hrs),':', num2str(mins)]);

% include LM comparison
figure; % by one testing data xTryGnl
subplot(2,3,1);
VisualizePredictionPlasmaSphere(ndtLM.ndt,inputLabelsGnl,xTryGnl,12,72,ndtLM.procFcnsInput,ndtLM.settingsXTrain,[], 'in General ndtLM')
subplot(2,3,2);
VisualizePredictionPlasmaSphere(ndtPolarLM.ndt,inputLabels,xTry,12,72,ndtPolarLM.procFcnsInput,ndtPolarLM.settingsXTrain,[], 'in Polar ndtLM')
subplot(2,3,3);
VisualizePredictionPlasmaSphere(ndtStatLPolarLM.ndt,inputLabels,xTry,12,72,ndtStatLPolarLM.procFcnsInput,ndtStatLPolarLM.settingsXTrain,ndtStatLPolarLM.Stat, 'in Stat Scaled Polar ndtLM')
subplot(2,3,4);
VisualizePredictionPlasmaSphere(nn45LM.nn45,inputLabelsGnl,xTryGnl,12,72,nn45LM.procFcnsInput,nn45LM.settingsXTrain,[], 'in General nn45LM')
sgtitle([DateStr,' ',num2str(hrs),':', num2str(mins)]);


% more comparison?
figure
subplot(2,3,1);
VisualizePredictionPlasmaSphere(nn45LM.nn45,inputLabelsGnl,xTryGnl,12,72,nn45LM.procFcnsInput,nn45LM.settingsXTrain,[], 'in General nn45LM')
subplot(2,3,4);
VisualizePredictionPlasmaSphere(nn45Model.nn45,inputLabelsGnl,xTryGnl,12,72,nn45Model.procFcnsInput,nn45Model.settingsXTrain,[],'in General nn45SGD')
sgtitle([DateStr,' ',num2str(hrs),':', num2str(mins)]);


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
VisualizePredictionPlasmaSphere(ndtPolar.ndt,inputLabels,xTry,12,36,ndtPolar.procFcnsInput,ndtPolar.settingsX,[], 'in ndtPolar')
subplot(2,3,5);
VisualizePredictionPlasmaSphere(nn45Polar.nn45,inputLabels,xTry,12,36,nn45Polar.procFcnsInput,nn45Polar.settingsXTrain,[], 'in nn45Polar')
subplot(2,3,1);
VisualizePredictionPlasmaSphere(ndtModel.ndt,inputLabelsGnl,xTryGnl,12,36,ndtModel.procFcnsInput,ndtModel.settingsX,[],'in General ndt')
subplot(2,3,4);
VisualizePredictionPlasmaSphere(nn45Model.nn45,inputLabelsGnl,xTryGnl,12,36,nn45Model.procFcnsInput,nn45Model.settingsXTrain,[],'in General nn45')
subplot(2,3,3);
VisualizePredictionPlasmaSphere(ndtStatLPolar.ndt,inputLabels,xTry,12,36,ndtStatLPolar.procFcnsInput,ndtStatLPolar.settingsXTrain,ndtStatLPolar.Stat, 'in ndtStatLPolar')
subplot(2,3,6);
VisualizePredictionPlasmaSphere(nn45StatPolar.nn45,inputLabels,xTry,12,36,nn45StatPolar.procFcnsInput,nn45StatPolar.settingsXTrain,nn45StatPolar.Stat, 'in nn45StatPolar')
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
