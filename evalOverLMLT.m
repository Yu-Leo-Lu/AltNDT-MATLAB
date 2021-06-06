startup

[PINE,trainIdx, testIdx] = loadPINE();
inputLabelsGnl = PINE.feature_names;
inputLabels = [PINE.feature_names(1:5); 'cmlt'; 'smlt'; PINE.feature_names(7:end)];
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

% ---------------------------- load models ----------------------------
% models
ndtStatLPolar = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_scaledStatLPolarMLT'));
ndtAdamStatLPolar = load(fullfile(dir, 'results','testModel', 'ndtAdam_50eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1_scaledStatLpolarMLT'));
ndt32StatLPolar = load(fullfile(dir, 'results', 'testModel', 'ndt32_50eps_lr1e-1_bs10000_mmt1e-1_scaledStatLPolarMLT'));
nn45StatLPolar = load(fullfile(dir, 'results', 'nn45_200eps_lr1e-1_bs10000_mmt1e-1_scaledStatLPolarMLT'));
ndt15AdamStatLPolar = load(fullfile(dir, 'results','testModel', 'ndt15Adam_50eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1_scaledStatLpolarMLT'));
ndt32AdamStatLPolar = load(fullfile(dir, 'results','testModel', 'ndt32Adam_50eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1_scaledStatLpolarMLT'));
ndt64AdamStatLPolar = load(fullfile(dir, 'results','testModel', 'ndt64Adam_50eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1_scaledStatLpolarMLT'));

% with polarMLT only
ndtPolar = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_polarMLT.mat'));
nn45Polar = load(fullfile(dir, 'results', 'nn45_200eps_lr1e-1_bs10000_mmt1e-1_polarMLT.mat'));
ndtAdamPolar = load(fullfile(dir, 'results','testModel', 'ndtAdam_20eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1_polarMLT'));
ndt15AdamPolar = load(fullfile(dir, 'results','testModel', 'ndt15Adam_200eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1_polarMLT'));
ndt15Polar = load(fullfile(dir, 'results','testModel', 'ndt15_neps_lr1e-1_bs10000_mmt9e-1_polarMLT'));

% with non-StatPolar
ndtModel = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat'));
nn45Model = load(fullfile(dir, 'results', 'nn45_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat'));

% LM models
nn45LM = load(fullfile(dir,'results','trainlm','nn45_40eps.mat'));
nn45StatLPolarLM = load(fullfile(dir, 'results','trainlm', 'nn45_40eps_scaledStatLPolarMLT.mat'));
ndtLM = load(fullfile(dir,'results','trainlm','ndt_40eps.mat'));
ndtPolarLM = load(fullfile(dir, 'results','trainlm', 'ndt_40eps_polarMLT.mat'));
ndtStatLPolarLM = load(fullfile(dir, 'results','trainlm', 'ndt_40eps_scaledStatLPolarMLT.mat'));
ndtStatWeightLPolarLM = load(fullfile(dir, 'results','trainlm', 'ndt_40eps_statLscaledWeightedPolarMLT'));

% ----------------- Principal Component plot -----------------
[XP1, X3]=loadSampleX(0.1);
XP1StdTitle=XP1{1}; XP1Std31=XP1{2}; XP1Std30=XP1{3};
X3StdTitle=X3{1}; X3Std31=X3{2}; X3Std30=X3{3};

% ------------------- debugging -------------------
iPC = 1;
figure; 
% subplot(3,1,1);
VisualizePcaByWeights(ndtStatWeightLPolarLM.ndt,inputLabels,XP1Std31{iPC+1},...
        12,72,ndtStatWeightLPolarLM.procFcnsInput,ndtStatWeightLPolarLM.settingsXTrain,ndtStatWeightLPolarLM.Stat,'',[],1)
% subplot(3,1,2);
% VisualizePredictionPlasmaSphere(ndtStatLPolarLM.ndt,inputLabels,xTry,...
%         12,72,ndtStatLPolarLM.procFcnsInput,ndtStatLPolarLM.settingsXTrain,ndtStatLPolarLM.Stat,'',[],0)
% subplot(3,1,3);
% VisualizePredictionPlasmaSphere(ndtPolarLM.ndt,inputLabels,xTry,...
%     12,72,ndtPolarLM.procFcnsInput,ndtPolarLM.settingsXTrain,[],'',[],0)


% TODO:
% make plot columns 1-5 PCs
% row 1-2, NDT -/+ DIFFERENCE 1 std
% row 3-4, PINE -/+ DIFFERENCE 1 std, 
% fix VisualizePca.m from VisualizePcaByWeights.m

% PINE vs WS NDT
figure; 
for iPC = 1:5
    for j = 1:2
            subplot(4,5,iPC);
            VisualizePcaByWeights(ndtStatWeightLPolarLM.ndt,inputLabels,XP1Std31{iPC+1}(1,:),...
                12,72,ndtStatWeightLPolarLM.procFcnsInput,ndtStatWeightLPolarLM.settingsXTrain,ndtStatWeightLPolarLM.Stat,['PC', num2str(iPC)],[],0)
            subplot(4,5,iPC+5);
             VisualizePcaByWeights(ndtStatWeightLPolarLM.ndt,inputLabels,XP1Std31{iPC+1}(2,:),...
                12,72,ndtStatWeightLPolarLM.procFcnsInput,ndtStatWeightLPolarLM.settingsXTrain,ndtStatWeightLPolarLM.Stat,'',[],0)
    end
    for j = 1:2
            subplot(4,5,iPC+10);
            VisualizePca(nn45LM.nn45,inputLabelsGnl,XP1Std30{iPC+1}(1,:),...
                12,72,nn45LM.procFcnsInput,nn45LM.settingsXTrain,[],'',[],0)
            subplot(4,5,iPC+15);
            VisualizePca(nn45LM.nn45,inputLabelsGnl,XP1Std30{iPC+1}(2,:),...
                12,72,nn45LM.procFcnsInput,nn45LM.settingsXTrain,[],'',[],0)
    end
end

% NDT: polar vs statPolar vs weightedStatPolar
figure; 
for iPC = 1:5
    for j = 1:2
            subplot(6,5,iPC);
            VisualizePcaByWeights(ndtStatWeightLPolarLM.ndt,inputLabels,XP1Std31{iPC+1}(1,:),...
                12,72,ndtStatWeightLPolarLM.procFcnsInput,ndtStatWeightLPolarLM.settingsXTrain,ndtStatWeightLPolarLM.Stat,['PC', num2str(iPC)],[],0)
            subplot(6,5,iPC+5);
             VisualizePcaByWeights(ndtStatWeightLPolarLM.ndt,inputLabels,XP1Std31{iPC+1}(2,:),...
                12,72,ndtStatWeightLPolarLM.procFcnsInput,ndtStatWeightLPolarLM.settingsXTrain,ndtStatWeightLPolarLM.Stat,'',[],0)
    end
    for j = 1:2
            subplot(6,5,iPC+10);
            VisualizePca(ndtStatLPolarLM.ndt,inputLabels,XP1Std31{iPC+1}(1,:),...
                12,72,ndtStatLPolarLM.procFcnsInput,ndtStatLPolarLM.settingsXTrain,ndtStatLPolarLM.Stat,'',[],0)
            subplot(6,5,iPC+15);
             VisualizePca(ndtStatLPolarLM.ndt,inputLabels,XP1Std31{iPC+1}(2,:),...
                12,72,ndtStatLPolarLM.procFcnsInput,ndtStatLPolarLM.settingsXTrain,ndtStatLPolarLM.Stat,'',[],0)
    end
    for j = 1:2
            subplot(6,5,iPC+20);
            VisualizePca(ndtPolarLM.ndt,inputLabels,XP1Std31{iPC+1}(1,:),...
                12,72,ndtPolarLM.procFcnsInput,ndtPolarLM.settingsXTrain,[],'',[],0)
            subplot(6,5,iPC+25);
             VisualizePca(ndtPolarLM.ndt,inputLabels,XP1Std31{iPC+1}(2,:),...
                12,72,ndtPolarLM.procFcnsInput,ndtPolarLM.settingsXTrain,[],'',[],0)
    end
end

% save color bar: just needed once
% figure; 
% VisualizePcaByWeights(ndtStatWeightLPolarLM.ndt,inputLabels,X1Std31{i+1}(1,:),...
%                 12,72,ndtStatWeightLPolarLM.procFcnsInput,ndtStatWeightLPolarLM.settingsXTrain,ndtStatWeightLPolarLM.Stat,['PC', num2str(i)],[],1)

% ----------------- Avg plot -----------------
xAvg31 = [nanmean(XTrain); nanmean(XTest); nanmean(X)];
xAvg30 = [nanmean(PINE.data_all.X(trainIdx,:)); nanmean(PINE.data_all.X(testIdx,:));nanmean(PINE.data_all.X)];
title={'Training Avg'; 'Testing Avg'; 'Overall Avg'};
figure; 
for iPC = 1:3
    subplot(4,3,iPC);
    VisualizePredictionPlasmaSphereByWeights(ndtStatWeightLPolarLM.ndt,inputLabels,xAvg31(iPC,:),...
        12,72,ndtStatWeightLPolarLM.procFcnsInput,ndtStatWeightLPolarLM.settingsXTrain,ndtStatWeightLPolarLM.Stat,title{iPC},[],0)
    subplot(4,3,iPC+3);
    VisualizePredictionPlasmaSphere(ndtStatLPolarLM.ndt,inputLabels,xAvg31(iPC,:),...
        12,72,ndtStatLPolarLM.procFcnsInput,ndtStatLPolarLM.settingsXTrain,ndtStatLPolarLM.Stat,'',[],0)
    subplot(4,3,iPC+3*2);
    VisualizePredictionPlasmaSphere(ndtPolarLM.ndt,inputLabels,xAvg31(iPC,:),...
        12,72,ndtPolarLM.procFcnsInput,ndtPolarLM.settingsXTrain,[],'',[],0)
    subplot(4,3,iPC+3*3);
    VisualizePredictionPlasmaSphere(nn45LM.nn45,inputLabelsGnl,xAvg30(iPC,:),...
        12,72,nn45LM.procFcnsInput,nn45LM.settingsXTrain,[],'',[],0)
end

% ---------------------- load 2001 data ----------------------
load(fullfile(dir, 'PINE_data', 'data2001'));
load(fullfile(dir, 'PINE_data', 'data2001Hourly'));
transition = 1.4;
                                                                         
DateStr = '26-Jun-2001'; hr = 19; min = 33;
days = datenum(DateStr) - datenum(2001,1,1);
xTryGnl = data2001(days*24*60+hr*60+min,:);
xTry = [xTryGnl(1:6),nan,xTryGnl(7:end)];
% xTryGnlH = data2001Hourly(days*24+hrs,:);
% xTryH = [xTryGnlH(1:6),nan,xTryGnlH(7:end)];
% xTryGstack = [xTryGnl;xTryGnlH];

DateStrs = {'26-Jun-2001','26-Jun-2001','26-Jun-2001','27-Jun-2001','27-Jun-2001'};
hrs = [11,19,22,00,11]; mins = [51,33,06,09,23];
xTryGnls = NaN(5,30); xTrys = NaN(5,31);
for iPC = 1:5
    DateStr = DateStrs{iPC}; hr = hrs(iPC); min = mins(iPC);
    days = datenum(DateStr) - datenum(2001,1,1);
    xTryGnl = data2001(days*24*60+hr*60+min,:);
    xTry = [xTryGnl(1:6),nan,xTryGnl(7:end)];
    xTryGnls(iPC,:) = xTryGnl; xTrys(iPC,:) = xTry;
end


% ------------------- compare polarMLT vs PINE -------------------

% figure; 
% for i = 1:5
%     timeTitle=[DateStrs{i},' ',num2str(hrs(i),'%02.f'),':', num2str(mins(i),'%02.f')];
%     subplot(2,5,i);
%     VisualizePredictionPlasmaSphere(ndtPolarLM.ndt,inputLabels,xTrys(i,:),...
%     12,72,ndtPolarLM.procFcnsInput,ndtPolarLM.settingsXTrain,[],timeTitle,[],0)
%     subplot(2,5,i+5);
%     VisualizePredictionPlasmaSphere(nn45LM.nn45,inputLabelsGnl,xTryGnls(i,:),...
%         12,72,nn45LM.procFcnsInput,nn45LM.settingsXTrain,[],'',[],0)
% end

% ----------------- WeightLPolarMLT,StatLPolarMLT, polarMLT, PINE -----------------
figure; 
for iPC = 1:5
    timeTitle=[DateStrs{iPC},' ',num2str(hrs(iPC),'%02.f'),':', num2str(mins(iPC),'%02.f')];
    subplot(4,5,iPC);
    VisualizePredictionPlasmaSphereByWeights(ndtStatWeightLPolarLM.ndt,inputLabels,xTrys(iPC,:),...
        12,72,ndtStatWeightLPolarLM.procFcnsInput,ndtStatWeightLPolarLM.settingsXTrain,ndtStatWeightLPolarLM.Stat,timeTitle,[],0)
    subplot(4,5,iPC+5);
    VisualizePredictionPlasmaSphere(ndtStatLPolarLM.ndt,inputLabels,xTrys(iPC,:),...
        12,72,ndtStatLPolarLM.procFcnsInput,ndtStatLPolarLM.settingsXTrain,ndtStatLPolarLM.Stat,'',[],0)
    subplot(4,5,iPC+5*2);
    VisualizePredictionPlasmaSphere(ndtPolarLM.ndt,inputLabels,xTrys(iPC,:),...
        12,72,ndtPolarLM.procFcnsInput,ndtPolarLM.settingsXTrain,[],'',[],0)
    subplot(4,5,iPC+5*3);
    VisualizePredictionPlasmaSphere(nn45LM.nn45,inputLabelsGnl,xTryGnls(iPC,:),...
        12,72,nn45LM.procFcnsInput,nn45LM.settingsXTrain,[],'',[],0)
end

% -----------------compare StatLPolarMLT, polarMLT vs PINE-----------------
figure; 
for iPC = 1:5
    timeTitle=[DateStrs{iPC},' ',num2str(hrs(iPC),'%02.f'),':', num2str(mins(iPC),'%02.f')];
    subplot(3,5,iPC);
    VisualizePredictionPlasmaSphere(ndtStatLPolarLM.ndt,inputLabels,xTrys(iPC,:),...
        12,72,ndtStatLPolarLM.procFcnsInput,ndtStatLPolarLM.settingsXTrain,ndtStatLPolarLM.Stat,timeTitle,[],0)
    subplot(3,5,iPC+5);
    VisualizePredictionPlasmaSphere(ndtPolarLM.ndt,inputLabels,xTrys(iPC,:),...
        12,72,ndtPolarLM.procFcnsInput,ndtPolarLM.settingsXTrain,[],'',[],0)
    subplot(3,5,iPC+5*2);
    VisualizePredictionPlasmaSphere(nn45LM.nn45,inputLabelsGnl,xTryGnls(iPC,:),...
        12,72,nn45LM.procFcnsInput,nn45LM.settingsXTrain,[],'',[],0)
end

% ----------------- get color bar plot -----------------
figure;
VisualizePredictionPlasmaSphere(ndtPolarLM.ndt,inputLabels,xTrys(iPC,:),...
    12,72,ndtPolarLM.procFcnsInput,ndtPolarLM.settingsXTrain,[],'',[],1)


% Stat Model NDT, SGD, ADAM and LM
figure;
subplot(2,3,1)
VisualizePredictionPlasmaSphere(ndtStatLPolar.ndt,inputLabels,xTry,...,
    12,72,ndtStatLPolar.procFcnsInput,ndtStatLPolar.settingsXTrain,...,
    ndtStatLPolar.Stat, 'in scaled Stat Polar SGD ndt',transition)
subplot(2,3,2)
VisualizePredictionPlasmaSphere(ndtAdamStatLPolar.ndt,inputLabels,xTry,...,
    12,72,ndtAdamStatLPolar.procFcnsInput,ndtAdamStatLPolar.settingsXTrain,...,
    ndtAdamStatLPolar.Stat, 'in scaled Stat Polar Adam ndt',transition)
subplot(2,3,3)
VisualizePredictionPlasmaSphere(ndtStatLPolarLM.ndt,inputLabels,xTry,...,
    12,72,ndtStatLPolarLM.procFcnsInput,ndtStatLPolarLM.settingsXTrain,...,
    ndtStatLPolarLM.Stat, 'in scaled Stat Polar LM ndt',transition)

subplot(2,3,4)
VisualizePredictionPlasmaSphere(ndt32StatLPolar.ndt,inputLabels,xTry,...,
    12,72,ndt32StatLPolar.procFcnsInput,ndt32StatLPolar.settingsXTrain,...,
    ndt32StatLPolar.Stat, 'in scaled Stat Polar SGD ndt32',transition)
subplot(2,3,5)
VisualizePredictionPlasmaSphere(ndt32AdamStatLPolar.ndt,inputLabels,xTry,...,
    12,72,ndt32AdamStatLPolar.procFcnsInput,ndt32AdamStatLPolar.settingsXTrain,...,
    ndt32AdamStatLPolar.Stat, 'in scaled Stat Polar Adam ndt32',transition)
subplot(2,3,6)
VisualizePredictionPlasmaSphere(ndt64AdamStatLPolar.ndt,inputLabels,xTry,...,
    12,72,ndt64AdamStatLPolar.procFcnsInput,ndt64AdamStatLPolar.settingsXTrain,...,
    ndt64AdamStatLPolar.Stat, 'in scaled Stat Polar Adam ndt64',transition)
sgtitle([DateStr,' ',num2str(hr),':', num2str(min)]);


% Stat Model test
figure;
subplot(2,3,1)
VisualizePredictionPlasmaSphere(ndtStatLPolar.ndt,inputLabels,xTry,...,
    12,72,ndtStatLPolar.procFcnsInput,ndtStatLPolar.settingsXTrain,...,
    ndtStatLPolar.Stat, 'in scaled Stat Polar SGD ndt',transition)
subplot(2,3,2)
VisualizePredictionPlasmaSphere(ndtAdamStatLPolar.ndt,inputLabels,xTry,...,
    12,72,ndtAdamStatLPolar.procFcnsInput,ndtAdamStatLPolar.settingsXTrain,...,
    ndtAdamStatLPolar.Stat, 'in scaled Stat Polar Adam ndt',transition)
subplot(2,3,4)
VisualizePredictionPlasmaSphere(ndt15AdamStatLPolar.ndt,inputLabels,xTry,...,
    12,72,ndt15AdamStatLPolar.procFcnsInput,ndt15AdamStatLPolar.settingsXTrain,...,
    ndt15AdamStatLPolar.Stat, 'in scaled Stat Polar Adam ndt15',transition)
subplot(2,3,5)
VisualizePredictionPlasmaSphere(ndt32AdamStatLPolar.ndt,inputLabels,xTry,...,
    12,72,ndt32AdamStatLPolar.procFcnsInput,ndt32AdamStatLPolar.settingsXTrain,...,
    ndt32AdamStatLPolar.Stat, 'in scaled Stat Polar Adam ndt32',transition)
sgtitle([DateStr,' ',num2str(hr),':', num2str(min)]);

% Model test
figure;
subplot(2,3,1)
VisualizePredictionPlasmaSphere(ndtAdamPolar.ndt,inputLabels,xTry,...
    12,72,ndtAdamPolar.procFcnsInput,ndtAdamPolar.settingsXTrain,...
    [], 'in Polar Adam ndt',transition)
subplot(2,3,2)
VisualizePredictionPlasmaSphere(ndt15AdamPolar.ndt,inputLabels,xTry,...
    12,72,ndt15AdamPolar.procFcnsInput,ndt15AdamPolar.settingsXTrain,...
    [], 'in Polar Adam ndt15',transition)
subplot(2,3,4)
VisualizePredictionPlasmaSphere(ndtPolar.ndt,inputLabels,xTry,...
    12,72,ndtPolar.procFcnsInput,ndtPolar.settingsXTrain,...
    [], 'in Polar SGD ndt',transition)
subplot(2,3,5)
VisualizePredictionPlasmaSphere(ndt15Polar.ndt,inputLabels,xTry,...
    12,72,ndt15Polar.procFcnsInput,ndt15Polar.settingsXTrain,[],...
    'in Polar SGD ndt15',transition)
sgtitle([DateStr,' ',num2str(hr),':', num2str(min)]);


% Stat Model test
figure;
subplot(2,3,1)
VisualizePredictionPlasmaSphere(ndtStatLPolar.ndt,inputLabels,xTry,12,72,ndtStatLPolar.procFcnsInput,ndtStatLPolar.settingsXTrain,ndtStatLPolar.Stat, 'in scaled Stat Polar SGD ndt',transition)
subplot(2,3,2)
VisualizePredictionPlasmaSphere(ndtAdamStatLPolar.ndt,inputLabels,xTry,12,72,ndtAdamStatLPolar.procFcnsInput,ndtAdamStatLPolar.settingsXTrain,ndtAdamStatLPolar.Stat, 'in scaled Stat Polar Adam ndt',transition)
subplot(2,3,4)
VisualizePredictionPlasmaSphere(ndt15AdamPolar.ndt,inputLabels,xTry,12,72,ndt15AdamPolar.procFcnsInput,ndt15AdamPolar.settingsXTrain,ndt15AdamPolar.Stat, 'in scaled Stat Polar Adam ndt15',transition)
subplot(2,3,5)
VisualizePredictionPlasmaSphere(ndt32AdamStatLPolar.ndt,inputLabels,xTry,12,72,ndt32AdamStatLPolar.procFcnsInput,ndt32AdamStatLPolar.settingsXTrain,ndt32AdamStatLPolar.Stat, 'in scaled Stat Polar Adam ndt32',transition)
sgtitle([DateStr,' ',num2str(hr),':', num2str(min)]);

% include LM comparison
figure; % by one testing data xTryGnl
subplot(2,3,1);
VisualizePredictionPlasmaSphere(ndtLM.ndt,inputLabelsGnl,xTryGnl,12,72,ndtLM.procFcnsInput,ndtLM.settingsXTrain,[], 'in General ndtLM',transition)
subplot(2,3,2);
VisualizePredictionPlasmaSphere(ndtPolarLM.ndt,inputLabels,xTry,12,72,ndtPolarLM.procFcnsInput,ndtPolarLM.settingsXTrain,[], 'in Polar ndtLM',transition)
subplot(2,3,3);
VisualizePredictionPlasmaSphere(ndtStatLPolarLM.ndt,inputLabels,xTry,12,72,ndtStatLPolarLM.procFcnsInput,ndtStatLPolarLM.settingsXTrain,ndtStatLPolarLM.Stat, 'in Stat Scaled Polar ndtLM',transition)
subplot(2,3,4);
VisualizePredictionPlasmaSphere(nn45LM.nn45,inputLabelsGnl,xTryGnl,12,72,nn45LM.procFcnsInput,nn45LM.settingsXTrain,[], 'in General nn45LM',transition)
sgtitle([DateStr,' ',num2str(hr),':', num2str(min)]);
subplot(2,3,6);
VisualizePredictionPlasmaSphere(nn45StatLPolarLM.nn45,inputLabels,xTry,12,72,nn45StatLPolarLM.procFcnsInput,nn45StatLPolarLM.settingsXTrain,nn45StatLPolarLM.Stat, 'in Stat Scaled Polar nn45LM',transition)
sgtitle([DateStr,' ',num2str(hr),':', num2str(min)]);


% more comparison?
figure
subplot(2,3,1);
VisualizePredictionPlasmaSphere(nn45LM.nn45,inputLabelsGnl,xTryGnl,12,72,nn45LM.procFcnsInput,nn45LM.settingsXTrain,[], 'in General nn45LM')
subplot(2,3,4);
VisualizePredictionPlasmaSphere(nn45Model.nn45,inputLabelsGnl,xTryGnl,12,72,nn45Model.procFcnsInput,nn45Model.settingsXTrain,[],'in General nn45SGD')
sgtitle([DateStr,' ',num2str(hr),':', num2str(min)]);


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
