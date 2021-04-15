startup
ndtModel = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat'));
ndt = ndtModel.ndt;
ndtInfo = ndtModel.ndtInfo;
load(fullfile(dir, 'results', 'nn45_40eps_lr_1e-1.mat'), 'nn45', 'nn45Info')
[PINE,trainIdx, testIdx] = loadPINE();
L = PINE.data_all.X(:,5);
MLT = PINE.data_all.X(:,6);
Density = PINE.data_all.t;

% RMSEscale = (settingst{2,1}.xmax - settingst{2,1}.xmin)/...
%     (settingst{2,1}.ymax - settingst{2,1}.ymin);

figure
hold on
% plot(ndttr.epoch, sqrt(perf_tree)*ones(size(ndttr.epoch)),'--','Color','#0072BD','LineWidth',2) %default color1
% plot(ndttr.epoch, sqrt(tperf_tree)*ones(size(ndttr.epoch)),'Color','#0072BD','LineWidth',2)
plot(1: size(ndtInfo.TrainingRMSE,2), ndtInfo.TrainingRMSE,'Color','#0072BD') % blue
plot(1: size(ndtInfo.ValidationRMSE,2), ndtInfo.ValidationRMSE,'x','Color','#0072BD','LineWidth',2) 
% plot(1: size(nn45Info.TrainingRMSE,2), nn45Info.TrainingRMSE*RMSEscale,'Color','#D95319') % orange
% plot(1: size(nn45Info.ValidationRMSE,2), nn45Info.ValidationRMSE*RMSEscale,'x','Color','#D95319','LineWidth',2)
legend('ndtTrain','ndtTest', 'nn45Train', 'nn45Test')
xlabel('Iterations')
ylabel('RMSE')
title('Performance Comparison in RMSE, lr=0.0001')
hold off

LTest = L(testIdx);
MLTTest = MLT(testIdx);
DensityTest = Density(testIdx);
StatTest=getPlasmaSphereStats(LTest,MLTTest,DensityTest);
VisualizePlasmaSPhereStats(StatTest, 'TestData')

% pre-process data
procFcnsInput = {}; procFcnsOutput = {};
procFcnsInput{1} = 'removeconstantrows'; procFcnsInput{2} = 'mapminmax';
procFcnsOutput{1} = 'removeconstantrows'; procFcnsOutput{2} = 'mapminmax';

% procFcnsOutput{1} = 'mapminmax';


[scaledX, settingsX] = preProcess(PINE.data_all.X, procFcnsInput);
[scaledt, settingst] = preProcess(PINE.data_all.t, procFcnsOutput);
RMSEscale = (settingst{2,1}.xmax - settingst{2,1}.xmin)/...
    (settingst{2,1}.ymax - settingst{2,1}.ymin);
XTest = scaledX(testIdx,:);
tTest = scaledt(testIdx, :);

% post-precess after the model
DensityNdt = predict(ndt, XTest);
DensityNdt = preProcessReverse(DensityNdt, procFcnsOutput, settingst);
StatNdt=getPlasmaSphereStats(LTest,MLTTest,DensityNdt);
VisualizePlasmaSPhereStats(StatNdt, 'NDT')

DensityNn45 = predict(nn45, XTest);
DensityNn45 = preProcessReverse(DensityNn45, procFcnsOutput, settingst);
StatNn45=getPlasmaSphereStats(LTest,MLTTest,DensityNn45);
VisualizePlasmaSPhereStats(StatNn45, 'NN45')
