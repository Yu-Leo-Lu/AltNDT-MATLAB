function [W0,b0,W1,b1,W2,b2,tree,error_check,K] = initAllWb(X,t,MaxNumSplits)

tree = fitrtree(X,t,'MaxNumSplits',MaxNumSplits);
K = floor(tree.NumNodes/2);
cutpt = tree.CutPoint;
cutpred = tree.CutPredictor;
cutpred = cutpred(~cellfun('isempty', cutpred));

W0 = zeros(K,size(X,2));
b0 = cutpt(~isnan(cutpt));
for i = 1:K
    W0(i,str2double(cutpred{i}(2:end))) = -1;
end

Children = tree.Children;
IsBranchNode = tree.IsBranchNode;
[W1, b1] = initW1b1(Children, IsBranchNode);

leafMean = tree.NodeMean(~IsBranchNode);
W2 = leafMean';
b2 = 0;

% for debug
preH = (W0*X'+b0)';
H = hardlim(preH);
preR = (W1*H'+b1)';
R = hardlim(preR);
ytreenet = (W2*R'+b2)';
ytree = predict(tree,X);

sH = satlin(preH);
preIR = (W1*sH'+b1)';
lR = logsig(preIR);
% msetree = mean((ytree-t).^2);
% msetreenet = mean((ytreenet-t).^2);
error_check = mean((ytreenet-ytree).^2);

end

