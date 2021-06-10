function [XPcaPortionPackage, XPackage]=loadSampleX(pcaPortion)
startup
Samples = load('SampleX.mat');

XStd = Samples.SampleX;
XStdTitle=cell(size(XStd));
XStd31 = cell(size(XStd));
XStd30 = cell(size(XStd));

XP1Std = cell(size(XStd));
XP1StdTitle=cell(size(XStd));
XP1Std31 = cell(size(XP1Std));
XP1Std30 = cell(size(XP1Std));

% generate sampleX title
for i = 1:length(XStd)
    if i == 1
        XStdTitle{i} = 'Overall Avg';
        XP1StdTitle{i} = 'Overall Avg';
    else
        XStdTitle{i}{1} = ['PC',num2str(i-1),' -1std'];
        XStdTitle{i}{2} = ['PC',num2str(i-1),' +1std'];
        XP1StdTitle{i}{1} = ['PC',num2str(i-1),['- ',num2str(pcaPortion),'std']];
        XP1StdTitle{i}{2} = ['PC',num2str(i-1),['- ',num2str(pcaPortion),'std']];
    end
end

% generate SampleX with 3 Std
% - polar, so each sample has 31 features
% - General, so each sample has 30 features
for i = 1:length(XStd)
    if i == 1
        row = XStd{i}';
        XStd31{i} = [row(1:4), nan,nan,nan, row(5:end)];
        XStd30{i} = [row(1:4), nan,nan, row(5:end)];
        XP1Std31{i} = [row(1:4), nan,nan,nan, row(5:end)];
        XP1Std30{i} = [row(1:4), nan,nan, row(5:end)];
    else
        row1 = XStd{i}(:,1)';
        row2 = XStd{i}(:,2)';
        Xbar = (row1+row2)/2;
        row1_PStd = Xbar-pcaPortion*(row1-row2);
        row2_PStd = Xbar+pcaPortion*(row1-row2);
        
        XStd31{i}(1,:) = [row1(1:4), nan,nan,nan, row1(5:end)];
        XStd31{i}(2,:) = [row2(1:4), nan,nan,nan, row2(5:end)];
        XStd30{i}(1,:) = [row1(1:4), nan,nan,    row1(5:end)];
        XStd30{i}(2,:) = [row2(1:4), nan,nan,    row2(5:end)];
        
        XP1Std31{i}(1,:) = [row1_PStd(1:4), nan,nan,nan, row1_PStd(5:end)];
        XP1Std31{i}(2,:) = [row2_PStd(1:4), nan,nan,nan, row2_PStd(5:end)];
        XP1Std30{i}(1,:) = [row1_PStd(1:4), nan,nan,    row1_PStd(5:end)];
        XP1Std30{i}(2,:) = [row2_PStd(1:4), nan,nan,    row2_PStd(5:end)];
    end
end

XPcaPortionPackage = {XP1StdTitle, XP1Std31, XP1Std30};
XPackage = {XStdTitle, XStd31, XStd30};

end