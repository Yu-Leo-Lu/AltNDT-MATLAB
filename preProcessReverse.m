function X = preProcessReverse(scaledX, procFcns, settings)
% params:
% scaledX: N x p scaled design matrix, N: # of data, p: # of features
% procFcns = {}
% procFcns{1} = 'removeconstantrows'; procFcns{2} = 'mapminmax';
% >> procFcns = {'removeconstantrows', 'mapminmax'};
% settings: output of preProcess.m function

% removeconstantrows: make sure no feature has the same values on all obvs
% mapminmax: make sure each feature max = 1 min = -1

% return:
% X: N x p reversed design matrix from scaledX, N: # of data, p: # of features
% using corresponding settings with procFcns.

scaledX = scaledX';
procFcnsLength = length(procFcns);

for i = procFcnsLength:-1:1
    f = str2func(procFcns{i});
    X = f('reverse',scaledX,settings{i});
    scaledX = X;
end

X = X';
end