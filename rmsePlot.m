function rmsePlot(ndt, nn45, methodStr)
% Zoom in box plot for performance comparison between NDT and NN45
start_epoch = 5;
is_zoomed = 0;

figure
hold on
% plot(ndtInfo.epoch, sqrt(perf_tree)*ones(size(ndtInfo.epoch)),'--','Color','#0072BD','LineWidth',2) %default color1
% plot(ndtInfo.epoch, sqrt(tperf_tree)*ones(size(ndtInfo.epoch)),'Color','#0072BD','LineWidth',2)
plot(ndt.ndtInfo.epoch(start_epoch:end), sqrt(ndt.ndtInfo.perf(start_epoch:end)),'--','Color','#D95319','LineWidth',2) %default color2
plot(ndt.ndtInfo.epoch(start_epoch:end), sqrt(ndt.ndtInfo.tperf(start_epoch:end)),'.-','MarkerSize',20,'Color','#D95319','LineWidth',2)
plot(nn45.nn45Info.epoch(start_epoch:end), sqrt(nn45.nn45Info.perf(start_epoch:end)),'--','Color','#0072BD','LineWidth',2) %default color1
plot(nn45.nn45Info.epoch(start_epoch:end), sqrt(nn45.nn45Info.tperf(start_epoch:end)),'.-','MarkerSize',20,'Color','#0072BD','LineWidth',2)
% legend('Tree,train','Tree,test','NDT,train','NDT,test','NN45,train','NN45,test')
legend({['NDT',methodStr, ', training'],['NDT',methodStr,', testing'],...
    ['PINE',methodStr,', training'],['PINE',methodStr,', testing']}, 'location', 'northeast', 'NumColumns',2)
xlabel('Epochs', 'FontSize',12)
ylabel('RMSE', 'FontSize', 12)
title('Performance Comparison in RMSE','FontSize',18)

if is_zoomed == 1
    % ----------------------zoom in box plot1------------------------------
    axes('position',[.625 .3 .25 .25])
    box on 
    indexOfInterest = (ndt.ndtInfo.epoch >=36) & (ndt.ndtInfo.epoch < 41);
    hold on
    plot(ndt.ndtInfo.epoch(indexOfInterest), sqrt(ndt.ndtInfo.perf(indexOfInterest)),'--','Color','#D95319','LineWidth',2) %default color2
    plot(ndt.ndtInfo.epoch(indexOfInterest), sqrt(ndt.ndtInfo.tperf(indexOfInterest)),'*-','Color','#D95319','LineWidth',2)
    plot(nn45.nn45Info.epoch(indexOfInterest), sqrt(nn45.nn45Info.perf(indexOfInterest)),'--','Color','#EDB120','LineWidth',2) %default color3
    plot(nn45.nn45Info.epoch(indexOfInterest), sqrt(nn45.nn45Info.tperf(indexOfInterest)),'*-','Color','#EDB120','LineWidth',2)
    % ----------------------zoom in box plot2------------------------------
    axes('position',[.225 .3 .25 .25])
    box on 
    indexOfInterest2 = (ndt.ndtInfo.epoch >=0) & (ndt.ndtInfo.epoch < 4);
    hold on
    plot(ndt.ndtInfo.epoch(indexOfInterest2), sqrt(ndt.ndtInfo.perf(indexOfInterest2)),'--','Color','#D95319','LineWidth',2) %default color2
    plot(ndt.ndtInfo.epoch(indexOfInterest2), sqrt(ndt.ndtInfo.tperf(indexOfInterest2)),'*-','Color','#D95319','LineWidth',2)
    plot(nn45.nn45Info.epoch(indexOfInterest2), sqrt(nn45.nn45Info.perf(indexOfInterest2)),'--','Color','#EDB120','LineWidth',2) %default color3
    plot(nn45.nn45Info.epoch(indexOfInterest2), sqrt(nn45.nn45Info.tperf(indexOfInterest2)),'*-','Color','#EDB120','LineWidth',2)
    % ----------------end of zoom in box plot-----------------------------
end
hold off

end