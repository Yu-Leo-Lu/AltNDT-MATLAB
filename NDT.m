% load PINE data
[PINE,train_idx, test_idx] = loadPINE();
X = PINE.data_all.X'; t = PINE.data_all.t';
% try to replace NAN data indices to a given data index
% then remove duplicate rows

% tree pre- and post- processing
procFcnsInput = {}; procFcnsOutput = {};
procFcnsInput{1} = 'removeconstantrows'; procFcnsInput{2} = 'mapminmax';
procFcnsOutput{1} = 'removeconstantrows'; procFcnsOutput{2} = 'mapminmax';
[Xp1, Xs1] = feval(procFcnsInput{1},X);
[Xp2, Xs2] = feval(procFcnsInput{2},Xp1);
[tp1, ts1] = feval(procFcnsOutput{1},t);
[tp2, ts2] = feval(procFcnsOutput{2},tp1);
% MaxNumSplits = 15, 24
MaxNumSplits = 25;

savedir = 'E:\Google Drive\A-projects\PINE\results';
bs = 30000;

for k = 2:2
    % initialize NDT
    [W0,b0,W1,b1,W2,b2,tree,error_check,K] = initAllWb(Xp2(:,PINE.inds_train{k})',tp2(:,PINE.inds_train{k})',MaxNumSplits);
%     view(tree,'Mode','graph')
%     ytree_train = predict(tree,Xrm(:,PINE.inds_train{k})')';
%     ytree_train = mapminmax('reverse', ytree_train, tPSrm);
%     ytree_train = removeconstantrows('reverse', ytree_train, tPSr);
%     mean((ytree_train-t(PINE.inds_train{k})).^2)
    
    ndt = feedforwardnet([K, K+1]);
    ndt.input.processFcns = procFcnsInput;
    ndt.output.processFcns = procFcnsOutput;
    ndt.divideFcn = 'divideind';
    ndt.divideParam.trainInd = PINE.inds_train{k};
    ndt.divideParam.valInd = PINE.inds_val{k};
    ndt.divideParam.testInd = PINE.inds_test;
    % ndt activation fcn input -> layer1 
    ndt.layers{1,1}.transferFcn = 'satlin'; 
    % ndt activation fcn layer1 -> layer 2
    ndt.layers{2,1}.transferFcn = 'logsig'; 
    ndt = configure(ndt,X,t);
    ndt.iw{1,1} = W0; ndt.lw{2,1} = W1; ndt.lw{3,2} = W2; % ndt weight initialization
    ndt.b{1} = b0; ndt.b{2} = b1; ndt.b{3} = b2;
    yndt_train = ndt(X(:,PINE.inds_train{k}));
%     mean((yndt_train-ytree_train).^2)
%     mean((yndt_train-t(:,PINE.inds_train{k})).^2)

    %train NDT
    ndt.trainParam.epochs = 40;
    
    [ndt, ndttr] = train(ndt, X, t);
    
    names = sprintf('ndt%d_val_%d.mat', MaxNumSplits, k);
    save(fullfile(savedir, names), 'ndt', 'ndttr')
end
