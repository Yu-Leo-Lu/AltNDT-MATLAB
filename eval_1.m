loaddir = 'E:\Google Drive\A-projects\PINE\results';
load(fullfile(loaddir, 'ndt25_val_1.mat'))
load(fullfile(loaddir, 'ndt25_val_2.mat'))
load(fullfile(loaddir, 'ndt15_val_3.mat'))
load(fullfile(loaddir, 'ndt15_val_4.mat'))
load(fullfile(loaddir, 'ndt15_val_5.mat'))

figure(2)
hold on
% plot(ndttr.epoch, sqrt(perf_tree)*ones(size(ndttr.epoch)),'--','Color','#0072BD','LineWidth',2) %default color1
% plot(ndttr.epoch, sqrt(tperf_tree)*ones(size(ndttr.epoch)),'Color','#0072BD','LineWidth',2)
plot(ndttr.epoch, sqrt(ndttr.perf),'--','Color','#D95319','LineWidth',2) %default color2
plot(ndttr.epoch, sqrt(ndttr.vperf),'-','Color','#D95319','LineWidth',2) %default color2
plot(ndttr.epoch, sqrt(ndttr.tperf),'x-','Color','#D95319','LineWidth',2)
% plot(nn45tr.epoch, sqrt(nn45tr.perf),'--','Color','#EDB120','LineWidth',2) %default color3
% plot(nn45tr.epoch, sqrt(nn45tr.tperf),'o-','Color','#EDB120','LineWidth',2)
% legend('Tree,train','Tree,test','NDT,train','NDT,test','NN45,train','NN45,test')
% legend('NDT,train','NDT,test','NN45,train','NN45,test')
legend('NDT,train','NDT,test')
xlabel('Epochs')
ylabel('RMSE')
title('Performance Comparison in RMSE')
hold off