[PINE, train_idx, test_idx] = loadPINE();
X = PINE.data_all.X'; t = PINE.data_all.t';

for k = 1:1
    nn45 = feedforwardnet(45);
    nn45 = configure(nn45,X,t);
    nn45.divideFcn = 'divideind';
    nn45.divideParam.trainInd = PINE.inds_train{k};
    nn45.divideParam.valInd = PINE.inds_val{k};
    nn45.divideParam.testInd = PINE.inds_test;
    
    %train
    nn45.trainParam.epochs = 40;
    [nn45, nn45tr] = train(nn45,X,t);
    names = sprintf('nn45_val_%d.mat', k);
    save(fullfile(savedir, names), 'nn45', 'nn45tr')
end

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
