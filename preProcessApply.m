function scaledX = preProcessApply(X,procFcns,settings)
% params:
% X: N x p design matrix, N: # of data, p: # of features
% procFcns = {}
% procFcns{1} = 'removeconstantrows'; procFcns{2} = 'mapminmax';
% >> procFcns = {'removeconstantrows', 'mapminmax'};

% removeconstantrows: make sure no feature has the same values on all obvs
% mapminmax: make sure each feature max = 1 min = -1
% settings: scaled settings for each procFcn in procFcns

% return: 
% scaledX: N x p scaled design matrix, N: # of data, p: # of features

X = X';
procFcnsLength = length(procFcns);

for i = 1:procFcnsLength
    f = str2func(procFcns{i});
    scaledX = f('apply',X, settings{i});
    X = scaledX;
end

scaledX = scaledX';
end