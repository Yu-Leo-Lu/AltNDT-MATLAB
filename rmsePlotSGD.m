function rmsePlotSGD(ndtMdl, nn45Mdl, iterPerEpoch, epochs, methodStr)
% Zoom in box plot for performance comparison between NDT and NN45


iterNdt = iterPerEpoch:numel(ndtMdl.ndtInfo.TrainingRMSE);
iterNdt = iterNdt(1:iterPerEpoch:end);
epsNdt = 1:numel(iterNdt);

iterNN45 = iterPerEpoch:numel(nn45Mdl.nn45Info.TrainingRMSE);
iterNN45 = iterNN45(1:iterPerEpoch:end);
epsNN45 = 1:numel(iterNN45);

TrainRmseNDTSgd = ndtMdl.ndtInfo.TrainingRMSE(iterPerEpoch:end);
TestRmseNDTSgd = ndtMdl.ndtInfo.ValidationRMSE(iterPerEpoch:end);
TrainRmseNDTSgdEps = TrainRmseNDTSgd(1:iterPerEpoch:end);
TestRmseNDTSgdEps = TestRmseNDTSgd(1:iterPerEpoch:end);

TrainRmseNN45Sgd = nn45Mdl.nn45Info.TrainingRMSE(iterPerEpoch:end);
TestRmseNN45Sgd = nn45Mdl.nn45Info.ValidationRMSE(iterPerEpoch:end);
TrainRmseNN45SgdEps = TrainRmseNN45Sgd(1:iterPerEpoch:end);
TestRmseNN45SgdEps = TestRmseNN45Sgd(1:iterPerEpoch:end);

if isempty(epochs) ~= 1
    epsNdt = epsNdt(epochs);
    epsNN45 = epsNN45(epochs);
    TrainRmseNDTSgdEps = TrainRmseNDTSgdEps(epochs);
    TestRmseNDTSgdEps = TestRmseNDTSgdEps(epochs);
    TrainRmseNN45SgdEps = TrainRmseNN45SgdEps(epochs);
    TestRmseNN45SgdEps = TestRmseNN45SgdEps(epochs);
end
if isempty(methodStr)==1
    methodStr = '';
end

figure
hold on
plot(epsNdt, TrainRmseNDTSgdEps,'--','Color','#D95319','LineWidth',2) %default color2
plot(epsNdt, TestRmseNDTSgdEps,'.-','MarkerSize',20,'Color','#D95319','LineWidth',2)
plot(epsNN45, TrainRmseNN45SgdEps,'--','Color','#0072BD','LineWidth',2) %default color1
plot(epsNN45, TestRmseNN45SgdEps,'.-','MarkerSize',20,'Color','#0072BD','LineWidth',2)

legend({['NDT',methodStr, ', training'],['NDT',methodStr,', testing'],...
    ['PINE',methodStr,', training'],['PINE',methodStr,', testing']}, 'location', 'northeast', 'NumColumns',2)
xlabel('Epochs', 'FontSize',12)
ylabel('RMSE', 'FontSize', 12)

title('Performance Comparison in RMSE','FontSize',18)


end