startup
% load(fullfile(dir, 'results', 'nn45_40eps_lr_1e-1.mat'), 'nn45', 'nn45Info')
[PINE,trainIdx, testIdx] = loadPINE();
X = PINE.data_all.X; t = PINE.data_all.t;
L = PINE.data_all.X(:,5);
MLT = PINE.data_all.X(:,6);
Density = PINE.data_all.t;
DensityTrain = Density(trainIdx);
DensityTest = PINE.data_all.t(testIdx);
polarMLT = zeros(size(MLT,1),2);
polarMLT(:,1) = cos(MLT*15*pi/180);
polarMLT(:,2) = sin(MLT*15*pi/180);
XPolar = [PINE.data_all.X(:,1:5), polarMLT, PINE.data_all.X(:,7:end)];


%pre- and post- processing
procFcnsInput = {}; procFcnsOutput = {};
procFcnsInput{1} = 'removeconstantrows'; procFcnsInput{2} = 'mapminmax';
procFcnsOutput{1} = 'removeconstantrows'; 
% procFcnsOutput{2} = 'mapminmax';

[XTrain, settingsX] = preProcess(X(trainIdx, :), procFcnsInput);
[yTrain, settingst] = preProcess(t(trainIdx, :), procFcnsOutput);

XTest = preProcessApply(X(testIdx, :),procFcnsInput,settingsX);
yTest = preProcessApply(t(testIdx, :),procFcnsOutput,settingst);

LTest = L(testIdx);
MLTTest = MLT(testIdx);
StatTest=getPlasmaSphereStats(LTest,MLTTest,yTest);
VisualizePlasmaSPhereStats(StatTest, 'TestData')

% in General, no polar MLT
ndtResult = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat'));
yNdt = predict(ndtResult.ndt, XTest);
StatNdt=getPlasmaSphereStats(LTest,MLTTest,yNdt);
VisualizePlasmaSPhereStats(StatNdt, 'NDT')

% - trainlm
ndtLM = load(fullfile(dir, 'results', 'trainlm', 'ndt_40eps'));
yNdtLM = ndtLM.ndt(XTest')';
% StatNdtLM=getPlasmaSphereStats(LTest,MLTTest,yNdtLM);
% VisualizePlasmaSPhereStats(StatNdtLM, 'NDTLM')
visualizeDensity(DensityTest, yNdtLM, LTest, MLTTest, Stat, 'by LM')

neLM = yNdtLM - DensityTest;
for iCell = 1:length(Stat)
    ind=find((LTest>=Stat(iCell).LRange(1))&(LTest<Stat(iCell).LRange(2))&...
            (MLTTest>=Stat(iCell).LonRange(1)/15)&(MLTTest<Stat(iCell).LonRange(2)/15));
    if length(ind)>1
        cellSTD = Stat(iCell).DensitySTD;
        neLM(ind) = neLM(ind)/cellSTD;
    end
end
figure
histogram(neLM);
title('Histogram of Normalization Error by LM');
figure
polarscatter(MLTTest*15*pi/180, LTest,[], neLM)
colorbar
title('Normalization Error by LM');


% PolarMLT:
ndtPolarResult = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_polarMLT.mat'));
[XTrainPolar, settingsX] = preProcess(XPolar(trainIdx, :), procFcnsInput);
XTestPolar = preProcessApply(XPolar(testIdx, :),procFcnsInput,settingsX);
yNdtPolar = predict(ndtPolarResult.ndt, XTestPolar);
% StatNdtPolar=getPlasmaSphereStats(LTest,MLTTest,yNdtPolar);
% VisualizePlasmaSPhereStats(StatNdtPolar, 'NDTPolar')
visualizeDensity(DensityTest, yNdtPolar, LTest, MLTTest, Stat, 'in Polar')


% - trainlm
ndtLMPolar = load(fullfile(dir, 'results', 'trainlm', 'ndt_40eps_polarMLT'));
yNdtPolarLM = ndtLMPolar.ndt(XTestPolar');
StatNdtPolarLM=getPlasmaSphereStats(LTest,MLTTest,yNdtPolarLM);
VisualizePlasmaSPhereStats(StatNdtPolarLM, 'NDTPolarLM')
