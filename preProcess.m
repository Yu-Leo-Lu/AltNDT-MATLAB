function [scaledX, settings] = preProcess(X,procFcns)
% params:
% X: N x p design matrix, N: # of data, p: # of features
% procFcns = {}
% procFcns{1} = 'removeconstantrows'; procFcns{2} = 'mapminmax';
% >> procFcns = {'removeconstantrows', 'mapminmax'};

% removeconstantrows: make sure no feature has the same values on all obvs
% mapminmax: make sure each feature max = 1 min = -1

% return: 
% scaledX: N x p scaled design matrix, N: # of data, p: # of features
% settings: scaled settings for each procFcn in procFcns
X = X';
procFcnsLength = length(procFcns);
settings = cell(procFcnsLength,1);

for i = 1:procFcnsLength
    f = str2func(procFcns{i});
    [scaledX, setting] = f(X);
    X = scaledX;
    settings{i} = setting;
end

scaledX = scaledX';
end