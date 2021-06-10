function rmsePlotSGD(ndtSGD, nn45SGD, iterPerEpoch)
% Zoom in box plot for performance comparison between NDT and NN45

iterNdt = iterPerEpoch:numel(ndtSGD.ndtInfo.TrainingRMSE);
iterNdt = iterNdt(1:iterPerEpoch:end);
epsNdt = 1:numel(iterNdt);

iterNN45 = iterPerEpoch:numel(nn45SGD.nn45Info.TrainingRMSE);
iterNN45 = iterNN45(1:iterPerEpoch:end);
epsNN45 = 1:numel(iterNN45);

TrainRmseNDTSgd = ndtSGD.ndtInfo.TrainingRMSE(iterPerEpoch:end);
TestRmseNDTSgd = ndtSGD.ndtInfo.ValidationRMSE(iterPerEpoch:end);
TrainRmseNDTSgdEps = TrainRmseNDTSgd(1:iterPerEpoch:end);
TestRmseNDTSgdEps = TestRmseNDTSgd(1:iterPerEpoch:end);

TrainRmseNN45Sgd = nn45SGD.nn45Info.TrainingRMSE(iterPerEpoch:end);
TestRmseNN45Sgd = nn45SGD.nn45Info.ValidationRMSE(iterPerEpoch:end);
TrainRmseNN45SgdEps = TrainRmseNN45Sgd(1:iterPerEpoch:end);
TestRmseNN45SgdEps = TestRmseNN45Sgd(1:iterPerEpoch:end);

figure
hold on
plot(epsNdt, TrainRmseNDTSgdEps,'--','Color','#D95319') %default color2
plot(epsNdt, TestRmseNDTSgdEps,'.-','MarkerSize',10,'Color','#D95319')
plot(epsNN45, TrainRmseNN45SgdEps,'--','Color','#EDB120') %default color3
plot(epsNN45, TestRmseNN45SgdEps,'.-','MarkerSize',10,'Color','#EDB120')

legend('NDT SGD, training','NDT SGD, testing',...
    'PINE SGD, training','PINE SGD, testing')
xlabel('Epochs')
ylabel('RMSE')
title('Performance Comparison in RMSE')


end