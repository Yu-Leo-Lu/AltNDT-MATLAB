startup

% load PINE data
[PINE,trainIdx, testIdx] = loadPINE();
X = PINE.data_all.X;
Density = PINE.data_all.t;

%pre- and post- processing
procFcnsInput = {}; 
% procFcnsOutput = {};
procFcnsInput{1} = 'removeconstantrows'; procFcnsInput{2} = 'mapminmax';
% procFcnsOutput{1} = 'removeconstantrows'; 
% procFcnsOutput{2} = 'mapminmax';

[XTrain, settingsXTrain] = preProcess(X(trainIdx, :), procFcnsInput);
% [yTrain, settingstTrain] = preProcess(Density(trainIdx, :), procFcnsOutput);
yTrain = Density(trainIdx);

XTest = preProcessApply(X(testIdx, :),procFcnsInput,settingsXTrain);
% yTest = preProcessApply(Density(testIdx, :),procFcnsOutput,settingstTrain);
yTest = Density(testIdx);

% -------------------------SGD training------------------------------------
% call ndgSGD:
ndtSGD

% fig = findall(groot,'Type','Figure');
% saveas(fig, fullfile(dir, 'figures', 'ndtTrainTest_lr_1e-1'));

% SGD saving
names = sprintf('ndt_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat');

save(fullfile(dir, 'results', names), 'ndt', 'ndtInfo',...
    'procFcnsInput', 'settingsXTrain')

% Adam saving
% names = sprintf('ndtAdam_50eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1.mat');
% save(fullfile(dir, 'results','testModel', names), 'ndt', 'ndtInfo',...
%      'procFcnsInput', 'settingsXTrain')


% call nn45SGD:
% nn45SGD
% names = sprintf('nn45_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat');
% 
% save(fullfile(dir, 'results', names), 'nn45', 'nn45Info',...
%     'procFcnsInput', 'settingsXTrain')

% -------------------------LM training-------------------------------------
% call ndtLM:
% ndtLM
% names = sprintf('ndt10_40eps.mat');
% save(fullfile(dir, 'results', 'trainlm', names), 'ndt', 'ndtInfo',...
%     'procFcnsInput', 'settingsXTrain')

% call nn45LM
% nn45LM
% 
% names = sprintf('nn45_40eps_scaledStatLPolarMLT.mat');
% save(fullfile(dir, 'results','trainlm', names), 'nn45', 'nn45Info',...
%     'procFcnsInput', 'settingsXTrain', 'Stat')

