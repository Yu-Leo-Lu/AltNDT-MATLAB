nn45Options = trainingOptions('sgdm', ...
    'miniBatchSize', 10000, ...
    'Momentum',0.1, ...
    'MaxEpochs',200,...
    'InitialLearnRate',1e-1, ...
    'ValidationData', {XTest, yTest},...
    'ValidationFrequency',293, ...
    'Verbose',true, ...
    'Plots','training-progress');


fc2 = fullyConnectedLayer(45);
fc3 = fullyConnectedLayer(1);


nn45Layers = [featureInputLayer(size(XTrain,2)),...
    fc2, sigmoidLayer,...
    fc3, regressionLayer];




[nn45, nn45Info] = trainNetwork(XTrain,yTrain,nn45Layers,nn45Options);

% fig = findall(groot,'Type','Figure');
% saveas(fig, fullfile(dir, 'figures', 'nn45TrainTest_lr_1e-1'));
% names = sprintf('nn45_40eps_lr_1e-1.mat');
% save(fullfile(dir, 'results', names), 'nn45', 'nn45Info')

yPredTest = predict(nn45,XTest);
rmseTest = sqrt(mean((yTest - yPredTest).^2));