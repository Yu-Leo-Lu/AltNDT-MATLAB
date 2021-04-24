startup

% load PINE data
[PINE,trainIdx, testIdx] = loadPINE();
X = PINE.data_all.X; t = PINE.data_all.t;

%pre- and post- processing
procFcnsInput = {}; procFcnsOutput = {};
procFcnsInput{1} = 'removeconstantrows'; procFcnsInput{2} = 'mapminmax';
procFcnsOutput{1} = 'removeconstantrows'; 
% procFcnsOutput{2} = 'mapminmax';

[XTrain, settingsXTrain] = preProcess(X(trainIdx, :), procFcnsInput);
% [yTrain, settingstTrain] = preProcess(t(trainIdx, :), procFcnsOutput);

XTest = preProcessApply(X(testIdx, :),procFcnsInput,settingsXTrain);
% yTest = preProcessApply(t(testIdx, :),procFcnsOutput,settingstTrain);

% call ndgSGD:
% ndtSGD

% fig = findall(groot,'Type','Figure');
% saveas(fig, fullfile(dir, 'figures', 'ndtTrainTest_lr_1e-1'));

% names = sprintf('ndt_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat');
% 
% save(fullfile(dir, 'results', names), 'ndt', 'ndtInfo',...
%     'procFcnsInput', 'settingsXTrain')

% call nn45SGD:
% nn45SGD
% names = sprintf('nn45_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat');
% 
% save(fullfile(dir, 'results', names), 'nn45', 'nn45Info',...
%     'procFcnsInput', 'settingsXTrain')


