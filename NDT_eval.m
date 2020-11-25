File   = 'PINE_train_val_test_dataset_IrinaOptimal.mat';
load(File); 

[PINE, train_idx, test_idx] = loadPINE();
X = PINE.data_all.X'; t = PINE.data_all.t';

load('nn45_40eps.mat')
load('ndt_40eps.mat')

% Evaluation
ynn45 = nn45(X(:,test_idx));
yndt = ndt(X(:,test_idx));
ytest = t(test_idx);
% mean((yndt-ytest).^2)

timet = data_all.time(test_idx);
startnum = datenum(2015,01,05);
endnum = datenum(2015,1,13);
s = find((timet>=startnum) & (timet<=endnum));

colorOrder = get(gca,'colororder');

figure(1)
hold on
plot(timet(s), ytest(s))
plot(timet(s), yndt(s))
plot(timet(s), ynn45(s))
datetick('x',2)
legend('target','NDT', 'NN45')
xlabel('Time')
ylabel('Plasma Density (log10)')
title('Performance Comparison in Plasma Density')
hold off

figure(2)
hold on
% plot(ndttr.epoch, sqrt(perf_tree)*ones(size(ndttr.epoch)),'--','Color','#0072BD','LineWidth',2) %default color1
% plot(ndttr.epoch, sqrt(tperf_tree)*ones(size(ndttr.epoch)),'Color','#0072BD','LineWidth',2)
plot(ndttr.epoch, sqrt(ndttr.perf),'--','Color','#D95319','LineWidth',2) %default color2
plot(ndttr.epoch, sqrt(ndttr.tperf),'Color','#D95319','LineWidth',2)
plot(nn45tr.epoch, sqrt(nn45tr.perf),'--','Color','#EDB120','LineWidth',2) %default color3
plot(nn45tr.epoch, sqrt(nn45tr.tperf),'Color','#EDB120','LineWidth',2)
% legend('Tree,train','Tree,test','NDT,train','NDT,test','NN45,train','NN45,test')
legend('NDT,train','NDT,test','NN45,train','NN45,test')
xlabel('Epochs')
ylabel('RMSE')
title('Performance Comparison in RMSE')
hold off

figure(3)
hold on
% plot(ndttr.epoch, sqrt(perf_tree)*ones(size(ndttr.epoch)),'--','Color','#0072BD','LineWidth',2) %default color1
% plot(ndttr.epoch, sqrt(tperf_tree)*ones(size(ndttr.epoch)),'Color','#0072BD','LineWidth',2)
plot(ndttr10.epoch, sqrt(ndttr10.perf),'--','Color','#D95319','LineWidth',2) %default color2
plot(ndttr10.epoch, sqrt(ndttr10.tperf),'Color','#D95319','LineWidth',2)
plot(nn45tr.epoch, sqrt(nn45tr.perf),'--','Color','#EDB120','LineWidth',2) %default color3
plot(nn45tr.epoch, sqrt(nn45tr.tperf),'Color','#EDB120','LineWidth',2)
% legend('Tree,train','Tree,test','NDT,train','NDT,test','NN45,train','NN45,test')
legend('NDT,train','NDT,test','NN45,train','NN45,test')
xlabel('Epochs')
ylabel('RMSE')
title('Performance Comparison in RMSE')
hold off

figure(4)
hold on
% plot(ndttr.epoch, sqrt(perf_tree)*ones(size(ndttr.epoch)),'--','Color','#0072BD','LineWidth',2) %default color1
% plot(ndttr.epoch, sqrt(tperf_tree)*ones(size(ndttr.epoch)),'Color','#0072BD','LineWidth',2)
plot(ndttr5.epoch, sqrt(ndttr5.perf),'--','Color','#D95319','LineWidth',2) %default color2
plot(ndttr5.epoch, sqrt(ndttr5.tperf),'Color','#D95319','LineWidth',2)
plot(nn45tr.epoch, sqrt(nn45tr.perf),'--','Color','#EDB120','LineWidth',2) %default color3
plot(nn45tr.epoch, sqrt(nn45tr.tperf),'Color','#EDB120','LineWidth',2)
% legend('Tree,train','Tree,test','NDT,train','NDT,test','NN45,train','NN45,test')
legend('NDT,train','NDT,test','NN45,train','NN45,test')
xlabel('Epochs')
ylabel('RMSE')
title('Performance Comparison in RMSE')
hold off

figure(5)
hold on
% plot(ndttr.epoch, sqrt(perf_tree)*ones(size(ndttr.epoch)),'--','Color','#0072BD','LineWidth',2) %default color1
% plot(ndttr.epoch, sqrt(tperf_tree)*ones(size(ndttr.epoch)),'Color','#0072BD','LineWidth',2)
plot(ndttr.epoch, sqrt(ndttr.perf),'--','Color',colorOrder(2,:),'LineWidth',2) %default color2
plot(ndttr.epoch, sqrt(ndttr.tperf),'Color',colorOrder(2,:),'LineWidth',2)
plot(nn45tr.epoch, sqrt(nn45tr.perf),'--','Color',colorOrder(3,:),'LineWidth',2) %default color3
plot(nn45tr.epoch, sqrt(nn45tr.tperf),'Color',colorOrder(3,:),'LineWidth',2)
plot(ndttr10.epoch, sqrt(ndttr10.perf),'--','Color',colorOrder(4,:),'LineWidth',2)
plot(ndttr10.epoch, sqrt(ndttr10.tperf),'Color',colorOrder(4,:),'LineWidth',2)
legend('NDT,train','NDT,test','NN45,train','NN45,test','NDT10,train', 'NDT10,test')
xlabel('Epochs')
ylabel('RMSE')
title('Performance Comparison in RMSE')
hold off