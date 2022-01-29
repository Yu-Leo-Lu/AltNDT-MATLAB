maxEps = 40;

Xp2 = X;
Xp2(trainIdx,:) = XTrain; Xp2(testIdx,:) = XTest;
Xp2 = Xp2';

tp2 = Density;
tp2(trainIdx) = yTrain; tp2(testIdx,:) = yTest;
tp2 = tp2';

nn45 = feedforwardnet(45);
nn45 = configure(nn45,Xp2,tp2);
nn45.input.processFcns = {};
nn45.output.processFcns = {};
nn45.divideFcn = 'divideind';
nn45.divideParam.trainInd = trainIdx;
nn45.divideParam.testInd = testIdx;

%train
nn45.trainParam.epochs = maxEps;
[nn45, nn45Info] = train(nn45,Xp2,tp2);
% names = sprintf('nn45_40eps.mat');
% 
% save(fullfile(dir, 'results','trainlm', names), 'nn45', 'nn45Info',...
%     'procFcnsInput', 'settingsXTrain')
% for i = 1:runs
%     nn53 = feedforwardnet(53);
%     nn53 = configure(nn53,X,t);
%     nn53.divideFcn = 'divideind';
%     nn53.divideParam.trainInd = train_idx;
%     nn53.divideParam.testInd = test_idx;
%     
%     %train
%     nn53.trainParam.epochs = 20;
%     [nn53, nn53tr] = train(nn53,X,t);
%     names = sprintf('nn53_20eps_%d.mat',i);
%     save(fullfile(savedir2, names), 'nn53', 'nn53tr')
% end
% 
% for i = 1:runs
%     nn38 = feedforwardnet(38);
%     nn38 = configure(nn38,X,t);
%     nn38.divideFcn = 'divideind';
%     nn38.divideParam.trainInd = train_idx;
%     nn38.divideParam.testInd = test_idx;
%     
%     %train
%     nn38.trainParam.epochs = 20;
%     [nn38, nn38tr] = train(nn38,X,t);
%     names = sprintf('nn38_20eps_%d.mat',i);
%     save(fullfile(savedir2, names), 'nn38', 'nn38tr')
% end
