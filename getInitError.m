MaxNumSplits = 10;
[W0,b0,W1,b1,W2,b2,tree,error_check,NumSplits] = ...
    initAllWb(XTrain,yTrain,MaxNumSplits);

sig = @(x)1./(1+exp(-x));
r1 = @(x)min(max(0,x),1);
indi = @(x)double(x>0); 

rmseTreeTrain = sqrt(mean((predict(tree, XTrain) - yTrain).^2))
rmseTreeTest = sqrt(mean((predict(tree, XTest) - yTest).^2));

l1 = r1(XTrain*W0'+[b0*ones(1,size(XTrain,1))]');
l2 = sig(l1*W1'+[b1*ones(1,size(l1,1))]');
yNdtTrain = l2*W2'+[b2*ones(1,size(l2,1))]';

rmseyNdtTrain = sqrt(mean((yNdtTrain - yTrain).^2))
rmseTreeTest = sqrt(mean((predict(tree, XTest) - yTest).^2));
