Xp2 = X;
Xp2(trainIdx,:) = XTrain; Xp2(testIdx,:) = XTest;
Xp2 = Xp2';

tp2 = Density;
tp2(trainIdx) = yTrain; tp2(testIdx,:) = yTest;
tp2 = tp2';

% MaxNumSplits = 15, 25
MaxNumSplits = 25;

% initialize NDT
[W0,b0,W1,b1,W2,b2,tree,error_check,K] = initAllWb(Xp2(:,trainIdx)',tp2(:,trainIdx)',MaxNumSplits);
%     view(tree,'Mode','graph')
%     ytree_train = predict(tree, XTrain)';
%     ytree_train = mapminmax('reverse', ytree_train, tPSrm);
%     ytree_train = removeconstantrows('reverse', ytree_train, tPSr);
%     mean((ytree_train-t(PINE.inds_train{k})).^2)

maxEps = 4;

ndt = feedforwardnet([K, K+1]);
ndt.input.processFcns = {};
ndt.output.processFcns = {};
ndt.divideFcn = 'divideind';
%     ndt.divideParam.trainInd = PINE.inds_train{k};
ndt.divideParam.trainInd = trainIdx;

%     ndt.divideParam.valInd = PINE.inds_val{k};
%     ndt.divideParam.testInd = PINE.inds_test;
ndt.divideParam.testInd = testIdx;

% ndt activation fcn input -> layer1 
ndt.layers{1,1}.transferFcn = 'satlin'; 
% ndt activation fcn layer1 -> layer 2
ndt.layers{2,1}.transferFcn = 'logsig'; 
ndt = configure(ndt,Xp2,tp2);
ndt.iw{1,1} = W0; ndt.lw{2,1} = W1; ndt.lw{3,2} = W2; % ndt weight initialization
ndt.b{1} = b0; ndt.b{2} = b1; ndt.b{3} = b2;
yndt_train = ndt(Xp2(:,trainIdx));
% mean((yndt_train-ytree_train).^2)
% mean((yndt_train-yTrain').^2)
% mean((ytree_train - yTrain').^2)

%train NDT
ndt.trainParam.epochs = maxEps;

[ndt, ndtInfo] = train(ndt, Xp2, tp2);


% names = sprintf('ndt%d_val_%d.mat', MaxNumSplits, k);
% save general version
% names = sprintf('ndt_40eps.mat');
% save(fullfile(dir, 'results', 'trainlm', names), 'ndt', 'ndtInfo','procFcnsInput','settingsXTrain')

% save polar version
% names = sprintf('ndt_40eps_polarMLT.mat');
% save(fullfile(dir, 'results', 'trainlm', names), 'ndt', 'ndtInfo','procFcnsInput','settingsXTrain')
% 

% ypred = ndt(Xp2(:, testIdx));
% rmse = sqrt(mean((tp2(testIdx) - ypred).^2))


