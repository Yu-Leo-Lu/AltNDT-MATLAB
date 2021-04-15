% MaxNumSplits = 15, 25
MaxNumSplits = 25;
[W0,b0,W1,b1,W2,b2,tree,error_check,NumSplits] = ...
    initAllWb(XTrain,yTrain,MaxNumSplits);


ndtOptions = trainingOptions('sgdm', ...
    'miniBatchSize', 10000, ...
    'Momentum',0.1, ...
    'MaxEpochs',200, ...
    'ValidationData', {XTest, yTest}, ...
    'ValidationFrequency',293, ...
    'InitialLearnRate',1e-1, ...
    'Verbose',true, ...
    'Plots','training-progress');

fc1 = fullyConnectedLayer(NumSplits);
fc1.Weights = W0;
fc1.Bias = b0;
fc2 = fullyConnectedLayer(NumSplits+1);
fc2.Weights = W1;
fc2.Bias = b1;
fc3 = fullyConnectedLayer(1);
fc3.Weights = W2;
fc3.Bias = b2;

ndtLayers = [featureInputLayer(size(XTrain,2)),...
    fc1, reluLayer,...
    fc2, sigmoidLayer,...
    fc3, regressionLayer];



[ndt, ndtInfo] = trainNetwork(XTrain,yTrain,ndtLayers,ndtOptions);

% fig = findall(groot,'Type','Figure');
% saveas(fig, fullfile(dir, 'figures', 'ndtTrainTest_lr_1e-1'));
% names = sprintf('ndt_40eps_lr_1e-1.mat');
% save(fullfile(dir, 'results', names), 'ndt', 'ndtInfo')




