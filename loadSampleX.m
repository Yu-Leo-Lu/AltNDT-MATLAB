function [X1, X3]=loadSampleX(pcaPortion)
startup
Samples = load('SampleX.mat');

X3Std = Samples.SampleX;
X3StdTitle=cell(size(X3Std));
X3Std31 = cell(size(X3Std));
X3Std30 = cell(size(X3Std));

X1Std = cell(size(X3Std));
X1StdTitle=cell(size(X3Std));
X1Std31 = cell(size(X1Std));
X1Std30 = cell(size(X1Std));

% generate sampleX title
for i = 1:length(X3Std)
    if i == 1
        X3StdTitle{i} = 'Overall Avg';
        X1StdTitle{i} = 'Overall Avg';
    else
        X3StdTitle{i}{1} = ['PC',num2str(i-1),' -3std'];
        X3StdTitle{i}{2} = ['PC',num2str(i-1),' +3std'];
        X1StdTitle{i}{1} = ['PC',num2str(i-1),['- ',num2str(pcaPortion),'std']];
        X1StdTitle{i}{2} = ['PC',num2str(i-1),['- ',num2str(pcaPortion),'std']];
    end
end

% generate SampleX with 3 Std
% - polar, so each sample has 31 features
% - General, so each sample has 30 features
for i = 1:length(X3Std)
    if i == 1
        row = X3Std{i};
        X3Std31{i} = [row(1:4), nan,nan,nan, row(5:end)];
        X3Std30{i} = [row(1:4), nan,nan, row(5:end)];
        X1Std31{i} = [row(1:4), nan,nan,nan, row(5:end)];
        X1Std30{i} = [row(1:4), nan,nan, row(5:end)];
    else
        row1 = X3Std{i}(1,:);
        row2 = X3Std{i}(2,:);
        Xbar = (row1+row2)/2;
        row1_1Std = Xbar-pcaPortion*(row1-row2)/3;
        row2_1Std = Xbar+pcaPortion*(row1-row2)/3;
        
        X3Std31{i}(1,:) = [row1(1:4), nan,nan,nan, row1(5:end)];
        X3Std31{i}(2,:) = [row2(1:4), nan,nan,nan, row2(5:end)];
        X3Std30{i}(1,:) = [row1(1:4), nan,nan,    row1(5:end)];
        X3Std30{i}(2,:) = [row2(1:4), nan,nan,    row2(5:end)];
        
        X1Std31{i}(1,:) = [row1_1Std(1:4), nan,nan,nan, row1_1Std(5:end)];
        X1Std31{i}(2,:) = [row2_1Std(1:4), nan,nan,nan, row2_1Std(5:end)];
        X1Std30{i}(1,:) = [row1_1Std(1:4), nan,nan,    row1_1Std(5:end)];
        X1Std30{i}(2,:) = [row2_1Std(1:4), nan,nan,    row2_1Std(5:end)];
    end
end

X1 = {X1StdTitle, X1Std31, X1Std30};
X3 = {X3StdTitle, X3Std31, X3Std30};

end