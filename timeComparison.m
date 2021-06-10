startup

[PINE,trainIdx, testIdx] = loadPINE();
inputLabelsGnl = PINE.feature_names;
inputLabels = [PINE.feature_names(1:5); 'cmlt'; 'smlt'; PINE.feature_names(7:end)];

ndt15LM = load(fullfile(dir,'results','trainlm', 'ndt15_40eps'));
ndt10LM = load(fullfile(dir,'results','trainlm', 'ndt10_40eps'));

ndtLM = load(fullfile(dir,'results','trainlm','ndt_40eps.mat'));
nn45LM = load(fullfile(dir,'results','trainlm','nn45_40eps.mat'));

ndtSGD = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat'));
ndtAdam = load(fullfile(dir,'results','testModel', 'ndtAdam_50eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1'));

figure
hold on
plot(ndtLM.ndtInfo.epoch, sqrt(ndtLM.ndtInfo.perf),'--','Color','#D95319','LineWidth',2) %default color2
plot(ndtLM.ndtInfo.epoch, sqrt(ndtLM.ndtInfo.tperf),'*-','Color','#D95319','LineWidth',2)
plot(nn45LM.nn45Info.epoch, sqrt(nn45LM.nn45Info.perf),'--','Color','#EDB120','LineWidth',2) %default color3
plot(nn45LM.nn45Info.epoch, sqrt(nn45LM.nn45Info.tperf),'*-','Color','#EDB120','LineWidth',2)
% legend('Tree,train','Tree,test','NDT,train','NDT,test','NN45,train','NN45,test')
legend('NDT LM train','NDT LM test','PINE, train','PINE, test')
xlabel('Epochs')
ylabel('RMSE')
title('Performance Comparison in RMSE')

sqrt(nn45LM.nn45Info.perf)
sqrt(nn45LM.nn45Info.tperf)

sqrt(ndtLM.ndtInfo.perf)
sqrt(ndtLM.ndtInfo.tperf)

sqrt(ndt15LM.ndtInfo.perf)
sqrt(ndt15LM.ndtInfo.tperf)

sqrt(ndt10LM.ndtInfo.perf)
sqrt(ndt10LM.ndtInfo.tperf)

min(find(ndtSGD.ndtInfo.TrainingRMSE<0.327))
min(find(ndtAdam.ndtInfo.TrainingRMSE<0.3170))


