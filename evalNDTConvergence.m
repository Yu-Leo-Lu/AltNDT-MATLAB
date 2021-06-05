startup

[PINE,trainIdx, testIdx] = loadPINE();
inputLabelsGnl = PINE.feature_names;
MLT = PINE.data_all.X(:,6); polarMLT = zeros(size(MLT,1),2);
polarMLT(:,1) = cos(MLT*15*pi/180);
polarMLT(:,2) = sin(MLT*15*pi/180);
inputLabels = [PINE.feature_names(1:5); 'cmlt'; 'smlt'; PINE.feature_names(7:end)];
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

ndtLM = load(fullfile(dir,'results','trainlm','ndt_40eps.mat'));
ndtPolarLM = load(fullfile(dir, 'results','trainlm', 'ndt_40eps_polarMLT.mat'));
ndtStatLPolarLM = load(fullfile(dir, 'results','trainlm', 'ndt_40eps_scaledStatLPolarMLT.mat'));
ndtWeightLPolarLM = load(fullfile(dir, 'results','trainlm', 'ndt_40eps_scaledWeightLPolarMLT'));
nn45LM = load(fullfile(dir,'results','trainlm', 'nn45_40eps.mat'));

XTestProcessed = preProcessApply(XTest, ...
    ndtWeightLPolarLM.procFcnsInput, ndtWeightLPolarLM.settingsXTrain);

yPredProcessed = ndtWeightLPolarLM.ndt(XTestProcessed')';
yPred = zeros(size(yPredProcessed));
statnDataPoint = [ndtWeightLPolarLM.Stat.nDataPoint];
statLRange = [Stat.LRange]';
for i = 1:length(yTest)
    iL = find(statLRange(:,1)<XTest(i,5) & statLRange(:,2)>XTest(i,5));
    yPred(i) = yPredProcessed(i)*sqrt(statnDataPoint(iL)/300);
end

sqrt(mean((yPred-yTest).^2))

figure
hold on
% plot(ndtLM.ndtInfo.epoch, sqrt(ndtLM.ndtInfo.perf),'--','Color','#D95319','LineWidth',2) %default color2
% plot(ndtLM.ndtInfo.epoch, sqrt(ndtLM.ndtInfo.tperf),'*-','Color','#D95319','LineWidth',2)
% plot(nn45LM.nn45Info.epoch, sqrt(nn45LM.nn45Info.perf),'--','Color','#EDB120','LineWidth',2) %default color3
% plot(nn45LM.nn45Info.epoch, sqrt(nn45LM.nn45Info.tperf),'*-','Color','#EDB120','LineWidth',2)
% plot(ndtPolarLM.ndttr.epoch, sqrt(ndtPolarLM.ndttr.perf),'--','Color','#7E2F8E','LineWidth',2) %default color3
% plot(ndtPolarLM.ndttr.epoch, sqrt(ndtPolarLM.ndttr.tperf),'*-','Color','#7E2F8E','LineWidth',2)
plot(ndtStatLPolarLM.ndtInfo.epoch, sqrt(ndtStatLPolarLM.ndtInfo.perf),'--','Color','#77AC30','LineWidth',2) %default color5
plot(ndtStatLPolarLM.ndtInfo.epoch, sqrt(ndtStatLPolarLM.ndtInfo.tperf),'*-','Color','#77AC30','LineWidth',2)
plot(ndtWeightLPolarLM.ndtInfo.epoch, sqrt(ndtWeightLPolarLM.ndtInfo.perf),'--','Color','#4DBEEE','LineWidth',2) %default color6
plot(ndtWeightLPolarLM.ndtInfo.epoch, sqrt(ndtWeightLPolarLM.ndtInfo.tperf),'*-','Color','#4DBEEE','LineWidth',2)
% legend('NDT, train','NDT, test',...
%        'PINE, train', 'PINE, test',...
%        'NDT polar, train', 'NDT polar, test', ', test')
legend('NDT Stat, train','NDT Stat, test',...
       'NDT Weighted, train', 'NDT Weighted, test', ', test')
   
xlabel('Epochs')
ylabel('RMSE')
title('Performance Comparison in RMSE')
