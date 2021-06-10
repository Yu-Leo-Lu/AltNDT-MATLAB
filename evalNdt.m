startup

[PINE,trainIdx, testIdx] = loadPINE();
inputLabelsGnl = PINE.feature_names;
inputLabels = [PINE.feature_names(1:5); 'cmlt'; 'smlt'; PINE.feature_names(7:end)];

nn45 = load(fullfile(dir,'results','trainlm', 'nn45_40eps.mat'));
ndt = load(fullfile(dir,'results','trainlm', 'ndt_40eps'));
ndt15 = load(fullfile(dir,'results','trainlm', 'ndt15_40eps'));
ndt10 = load(fullfile(dir,'results','trainlm', 'ndt10_40eps'));
ndtSGD = load(fullfile(dir,'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc'));
ndtAdam = load(fullfile(dir,'results','testModel', 'ndtAdam_50eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1'));
nn45SGD = load(fullfile(dir,'results', 'nn45_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc'));

[ndtAIC, ndtBIC] = infoCrit(ndtInfo.tperf(end), length(ndtInfo.trainInd), ...
    length(getwb(ndt.ndt)));
[nn45AIC, nn45BIC] = infoCrit(nn45Info.tperf(end), length(nn45Info.trainInd), ...
    length(getwb(nn45)));


%---------------------- RMSE plot convergence SGD ----------------------
rmsePlotSGD(ndtSGD, nn45SGD, 293)
rmsePlotSGD(ndtAdam, nn45SGD, 293)

% ---------------------- load 2001 data ----------------------
load(fullfile(dir, 'PINE_data', 'data2001'));
load(fullfile(dir, 'PINE_data', 'data2001Hourly'));
transition = 1.4;
                                                                         
DateStr = '26-Jun-2001'; hr = 19; min = 33;
days = datenum(DateStr) - datenum(2001,1,1);
xTryGnl = data2001(days*24*60+hr*60+min,:);
xTry = [xTryGnl(1:6),nan,xTryGnl(7:end)];
% xTryGnlH = data2001Hourly(days*24+hrs,:);
% xTryH = [xTryGnlH(1:6),nan,xTryGnlH(7:end)];
% xTryGstack = [xTryGnl;xTryGnlH];

DateStrs = {'26-Jun-2001','26-Jun-2001','26-Jun-2001','27-Jun-2001','27-Jun-2001'};
hrs = [11,19,22,00,11]; mins = [51,33,06,09,23];
xTryGnls = NaN(5,30); xTrys = NaN(5,31);
for i = 1:5
    DateStr = DateStrs{i}; hr = hrs(i); min = mins(i);
    days = datenum(DateStr) - datenum(2001,1,1);
    xTryGnl = data2001(days*24*60+hr*60+min,:);
    xTry = [xTryGnl(1:6),nan,xTryGnl(7:end)];
    xTryGnls(i,:) = xTryGnl; xTrys(i,:) = xTry;
end

% ----------------- NDT25, 15, 10 vs PINE on 2001 -----------------
figure; 
for i = 1:5
    timeTitle=[DateStrs{i},' ',num2str(hrs(i),'%02.f'),':', num2str(mins(i),'%02.f')];
    subplot(4,5,i);
    VisualizePredictionPlasmaSphere(ndt.ndt,inputLabelsGnl,xTryGnls(i,:),...
        12,72,ndt.procFcnsInput,ndt.settingsXTrain,[],timeTitle,[],0)
    subplot(4,5,i+5);
    VisualizePredictionPlasmaSphere(ndt15.ndt,inputLabelsGnl,xTryGnls(i,:),...
        12,72,ndt15.procFcnsInput,ndt15.settingsXTrain,[],'',[],0)
    subplot(4,5,i+5*2);
    VisualizePredictionPlasmaSphere(ndt10.ndt,inputLabelsGnl,xTryGnls(i,:),...
        12,72,ndt10.procFcnsInput,ndt10.settingsXTrain,[],'',[],0)
    subplot(4,5,i+5*3);
    VisualizePredictionPlasmaSphere(nn45.nn45,inputLabelsGnl,xTryGnls(i,:),...
        12,72,nn45.procFcnsInput,nn45.settingsXTrain,[],'',[],0)
end

% ---------------------- Evaluation ----------------------
ynn45 = nn45(X(:,test_idx));
yndt = ndt.ndt(X(:,test_idx));
ytest = Density(test_idx);
% mean((yndt-ytest).^2)

timet = PINE.data_all.time(test_idx);
% startnum = datenum(2015,06,07);
startnum = datenum(2015,08,01);
endnum = datenum(2015,08,09);
% endnum = datenum(2015,8,16);
s = find((timet>=startnum) & (timet<=endnum));
colorOrder = get(gca,'colororder');

figure(1)
hold on
plot(timet(s), ytest(s))
plot(timet(s), yndt(s))
plot(timet(s), ynn45(s))
datetick('x',6)
legend('target','NDT', 'NN45')
xlabel('Time')
ylabel('Plasma Density (log10)')
title('Performance Comparison in Plasma Density')
hold off

% RMSE plot with box zoom in
rmsePlot(ndt, nn45, 'NDT')
length(getwb(ndt.ndt))

rmsePlot(ndt15, nn45, 'NDT15')
length(getwb(ndt15.ndt))

rmsePlot(ndt10, nn45, 'NDT10')
length(getwb(ndt10.ndt))
length(getwb(nn45.nn45))


% ---------------------- old plot ----------------------
figure(4)
hold on
% plot(ndtInfo.epoch, sqrt(perf_tree)*ones(size(ndtInfo.epoch)),'--','Color','#0072BD','LineWidth',2) %default color1
% plot(ndtInfo.epoch, sqrt(tperf_tree)*ones(size(ndtInfo.epoch)),'Color','#0072BD','LineWidth',2)
plot(ndtInfo5.epoch, sqrt(ndtInfo5.perf),'--','Color','#D95319','LineWidth',2) %default color2
plot(ndtInfo5.epoch, sqrt(ndtInfo5.tperf),'Color','#D95319','LineWidth',2)
plot(nn45Info.epoch, sqrt(nn45Info.perf),'--','Color','#EDB120','LineWidth',2) %default color3
plot(nn45Info.epoch, sqrt(nn45Info.tperf),'Color','#EDB120','LineWidth',2)
% legend('Tree,train','Tree,test','NDT,train','NDT,test','NN45,train','NN45,test')
legend('NDT,train','NDT,test','NN45,train','NN45,test')
xlabel('Epochs')
ylabel('RMSE')
title('Performance Comparison in RMSE')
hold off

figure(5)
hold on
% plot(ndtInfo.epoch, sqrt(perf_tree)*ones(size(ndtInfo.epoch)),'--','Color','#0072BD','LineWidth',2) %default color1
% plot(ndtInfo.epoch, sqrt(tperf_tree)*ones(size(ndtInfo.epoch)),'Color','#0072BD','LineWidth',2)
plot(ndtInfo.epoch, sqrt(ndtInfo.perf),'--','Color',colorOrder(2,:),'LineWidth',2) %default color2
plot(ndtInfo.epoch, sqrt(ndtInfo.tperf),'Color',colorOrder(2,:),'LineWidth',2)
plot(nn45Info.epoch, sqrt(nn45Info.perf),'--','Color',colorOrder(3,:),'LineWidth',2) %default color3
plot(nn45Info.epoch, sqrt(nn45Info.tperf),'Color',colorOrder(3,:),'LineWidth',2)
plot(ndtInfo10.epoch, sqrt(ndtInfo10.perf),'--','Color',colorOrder(4,:),'LineWidth',2)
plot(ndtInfo10.epoch, sqrt(ndtInfo10.tperf),'Color',colorOrder(4,:),'LineWidth',2)
legend('NDT,train','NDT,test','NN45,train','NN45,test','NDT10,train', 'NDT10,test')
xlabel('Epochs')
ylabel('RMSE')
title('Performance Comparison in RMSE')
hold off