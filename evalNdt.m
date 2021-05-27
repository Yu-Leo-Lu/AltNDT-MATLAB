startup
dir = 'E:\Google Drive\A-projects\PINE';
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

load('PINE_train_val_test_dataset_IrinaOptimal.mat'); 
[PINE, train_idx, test_idx] = loadPINE();
X = PINE.data_all.X'; Density = PINE.data_all.t';


% Evaluation
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