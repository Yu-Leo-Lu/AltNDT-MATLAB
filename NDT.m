% load PINE data
[PINE, train_idx, test_idx] = loadPINE();
X = PINE.data_all.X'; t = PINE.data_all.t';

% tree pre- and post- processing
procFcnsInput = {}; procFcnsOutput = {};
procFcnsInput{1} = 'removeconstantrows'; procFcnsInput{2} = 'mapminmax';
procFcnsOutput{1} = 'removeconstantrows'; procFcnsOutput{2} = 'mapminmax';
[Xp1, Xs1] = feval(procFcnsInput{1},X);
[Xp2, Xs2] = feval(procFcnsInput{2},Xp1);
[tp1, ts1] = feval(procFcnsOutput{1},t);
[tp2, ts2] = feval(procFcnsOutput{2},tp1);
MaxNumSplits = 5;

runs = 1;
for i = 1:runs
    % initialize NDT
    [W0,b0,W1,b1,W2,b2,tree,error_check,K] = initAllWb(Xp2(:,train_idx)',tp2(:,train_idx)',MaxNumSplits);
%     view(tree,'Mode','graph')
%     ytree_train = predict(tree,Xrm(:,train_idx)')';
%     ytree_train = mapminmax('reverse', ytree_train, tPSrm);
%     ytree_train = removeconstantrows('reverse', ytree_train, tPSr);
%     mean((ytree_train-t(train_idx)).^2)
    
    ndt = feedforwardnet([K, K+1]);
    ndt.input.processFcns = procFcnsInput;
    ndt.output.processFcns = procFcnsOutput;
    ndt.divideFcn = 'divideind';
    ndt.divideParam.trainInd = train_idx;
    ndt.divideParam.testInd = test_idx;
    ndt.layers{1,1}.transferFcn = 'satlin'; % ndt activation fcn input -> layer1
    ndt.layers{2,1}.transferFcn = 'logsig'; % ndt activation fcn layer1-> layer 2
    ndt = configure(ndt,X,t);
    ndt.iw{1,1} = W0; ndt.lw{2,1} = W1; ndt.lw{3,2} = W2; % ndt weight initialization
    ndt.b{1} = b0; ndt.b{2} = b1; ndt.b{3} = b2;
    yndt_train = ndt(X(:,train_idx));
%     mean((yndt_train-ytree_train).^2)
%     mean((yndt_train-t(:,train_idx)).^2)

    %train NDT
    ndt.trainParam.epochs = 40;
    [ndt, ndttr] = train(ndt, X, t);
    
    % save('ndt_20eps.mat', 'ndt', 'ndttr')
    save(sprintf('ndt5_alltrain40eps_%d.mat',i), 'ndt', 'ndttr')
end