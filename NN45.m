[PINE, train_idx, test_idx] = loadPINE();
X = PINE.data_all.X'; t = PINE.data_all.t';

runs = 10;
for i = 5:runs
    nn45 = feedforwardnet(45);
    nn45 = configure(nn45,X,t);
    nn45.divideFcn = 'divideind';
    nn45.divideParam.trainInd = train_idx;
    nn45.divideParam.testInd = test_idx;
    
    %train
    nn45.trainParam.epochs = 20;
    [nn45, nn45tr] = train(nn45,X,t);
    save(sprintf('nn45_20eps_%d.mat',i), 'nn45', 'nn45tr')
end