startup

% load PINE data
[PINE,trainIdx, testIdx] = loadPINE();
MLT = PINE.data_all.X(:,6); polarMLT = zeros(size(MLT,1),2);
polarMLT(:,1) = cos(MLT*15*pi/180);
polarMLT(:,2) = sin(MLT*15*pi/180);
X = [PINE.data_all.X(:,1:5), polarMLT, PINE.data_all.X(:,7:end)];
t = PINE.data_all.t;

%pre- and post- processing
procFcnsInput = {}; procFcnsOutput = {};
procFcnsInput{1} = 'removeconstantrows'; procFcnsInput{2} = 'mapminmax';
procFcnsOutput{1} = 'removeconstantrows'; 
% procFcnsOutput{2} = 'mapminmax';

[XTrain, settingsXTrain] = preProcess(X(trainIdx, :), procFcnsInput);
% [yTrain, settingst] = preProcess(t(trainIdx, :), procFcnsOutput);

XTest = preProcessApply(X(testIdx, :),procFcnsInput,settingsXTrain);
% yTest = preProcessApply(t(testIdx, :),procFcnsOutput,settingst);

% call ndgSGD:
% ndtSGD
% 
names = sprintf('ndt_200eps_lr1e-1_bs10000_mmt1e-1_polarMLT.mat');
save(fullfile(dir, 'results', names), 'ndt', 'ndtInfo',...
    'procFcnsInput', 'settingsXTrain')

% call nn45:
% nn45SGD
% names = sprintf('nn45_200eps_lr1e-1_bs10000_mmt1e-1_polarMLT.mat');
% 
% save(fullfile(dir, 'results', names), 'nn45', 'nn45Info',...
%     'procFcnsInput', 'settingsXTrain')

% call ndtLM:
% ndtLM
% 
% names = sprintf('ndt_40eps_polarMLT.mat');
% save(fullfile(dir, 'results','trainlm', names), 'ndt', 'ndtInfo',...
%     'procFcnsInput', 'settingsXTrain')

% call nn45LM:
% nn45LM
% 
% names = sprintf('nn45_40eps_polarMLT.mat');
% save(fullfile(dir, 'results','trainlm', names), 'nn45', 'nn45Info',...
%     'procFcnsInput', 'settingsXTrain')
