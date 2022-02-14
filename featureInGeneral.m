startup

% load PINE data
% [PINE,trainIdx, testIdx] = loadPINE();

% load sequential PINE data
[PINE,inds_train, inds_test] = loadPineTimeSeq();
trainIdx = inds_train{1};
testIdx = inds_test{1};

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
% ndtSGD
disp('-------------------- ndt SGD starts --------------------')
MaxNumSplits = 25;
tic; ndtSGD; toc;
disp('-------------------- ndt SGD ends --------------------')

% fig = findall(groot,'Type','Figure');
% saveas(fig, fullfile(dir, 'figures', 'ndtTrainTest_lr_1e-1'));

% SGD saving
% names = sprintf('ndt_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat');
% 
% save(fullfile(dir, 'results', names), 'ndt', 'ndtInfo',...
%     'procFcnsInput', 'settingsXTrain')

% Adam 
disp('-------------------- ndt Adam starts --------------------')
MaxNumSplits = 25;
maxEps = 40;
tic; ndtSGD; toc;
disp('-------------------- ndt Adam ends --------------------')
% names = sprintf('ndtAdam_40eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1.mat');
% save(fullfile(dir, 'results','testModel','timeSeq', names), 'ndt', 'ndtInfo',...
%      'procFcnsInput', 'settingsXTrain')


% call nn45SGD:
% nn45SGD
disp('-------------------- PINE SGD starts --------------------')
tic; nn45SGD; toc;
disp('-------------------- PINE SGD ends --------------------')
% names = sprintf('nn45_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat');
% 
% save(fullfile(dir, 'results', names), 'nn45', 'nn45Info',...
%     'procFcnsInput', 'settingsXTrain')

disp('-------------------- PINE Adam starts --------------------')
maxEps = 40;
tic; nn45SGD; toc;
disp('-------------------- PINE Adam ends --------------------')

% names = sprintf('nn45_40eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1.mat');
% 
% save(fullfile(dir, 'results','timeSeq', names), 'nn45', 'nn45Info',...
%     'procFcnsInput', 'settingsXTrain')


% -----------------------------Time Sequential-----------------------------
% Adam 
disp('-------------------- ndt Adam starts --------------------')
MaxNumSplits = 25;
maxEps = 40;
tic; ndtSGD; toc;
disp('-------------------- ndt Adam ends --------------------')
names = sprintf('ndtAdam_40eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1.mat');
save(fullfile(dir, 'results','Adam','timeseq', names), 'ndt', 'ndtInfo',...
     'procFcnsInput', 'settingsXTrain')

disp('-------------------- PINE Adam starts --------------------')
maxEps = 40;
tic; nn45SGD; toc;
disp('-------------------- PINE Adam ends --------------------')
names = sprintf('nn45_40eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1.mat');
save(fullfile(dir, 'results','Adam','timeseq', names), 'nn45', 'nn45Info',...
    'procFcnsInput', 'settingsXTrain')



% -------------------------LM training-------------------------------------
% Indices from Irina
% call ndtLM:
disp('-------------------- ndt LM starts --------------------')
MaxNumSplits = 25;
tic; ndtLM; toc;
disp('-------------------- ndt LM ends --------------------')

names = sprintf('ndt25_40eps.mat');
save(fullfile(dir, 'results', 'trainlm', names), 'ndt', 'ndtInfo',...
    'procFcnsInput', 'settingsXTrain')

% call nn45LM
disp('-------------------- PINE LM starts --------------------')
tic; nn45LM; toc;
disp('-------------------- PINE LM ends --------------------')
names = sprintf('nn45_40eps.mat');
save(fullfile(dir, 'results','trainlm', names), 'nn45', 'nn45Info',...
    'procFcnsInput', 'settingsXTrain')

% Time sequential seperated indices
% call ndtLM:
disp('-------------------- ndt LM starts --------------------')
MaxNumSplits = 25;
tic; ndtLM; toc;
disp('-------------------- ndt LM ends --------------------')

% names = sprintf('ndt25_40eps_timesep.mat');
% save(fullfile(dir, 'results', 'trainlm','TimeSeq', names), 'ndt', 'ndtInfo',...
%     'procFcnsInput', 'settingsXTrain')

% call nn45LM
disp('-------------------- PINE LM starts --------------------')
tic; nn45LM; toc;
disp('-------------------- PINE LM ends --------------------')
names = sprintf('nn45_40eps_timesep.mat');
save(fullfile(dir, 'results','trainlm','TimeSeq', names), 'nn45', 'nn45Info',...
    'procFcnsInput', 'settingsXTrain')

