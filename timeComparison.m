startup

[PINE,trainIdx, testIdx] = loadPINE();
inputLabelsGnl = PINE.feature_names;
inputLabels = [PINE.feature_names(1:5); 'cmlt'; 'smlt'; PINE.feature_names(7:end)];

ndt15LM = load(fullfile(dir,'results','trainlm', 'ndt15_40eps'));
ndt10LM = load(fullfile(dir,'results','trainlm', 'ndt10_40eps'));

ndtLmMld = load(fullfile(dir,'results','trainlm','ndt_40eps.mat'));
nn45LmMdl = load(fullfile(dir,'results','trainlm','nn45_40eps.mat'));

ndtSGDMdl = load(fullfile(dir, 'results', 'ndt_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc.mat'));
nn45SGDMdl = load(fullfile(dir,'results', 'nn45_200eps_lr1e-1_bs10000_mmt1e-1_tnoproc'));
ndtAdamMdl = load(fullfile(dir,'results','testModel', 'ndtAdam_50eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1'));

rmsePlotSGD(ndtSGDMdl, nn45SGDMdl, 293)

figure
hold on
plot(ndtLmMld.ndtInfo.epoch, sqrt(ndtLmMld.ndtInfo.perf),'--','Color','#D95319','LineWidth',2) %default color2
plot(ndtLmMld.ndtInfo.epoch, sqrt(ndtLmMld.ndtInfo.tperf),'*-','Color','#D95319','LineWidth',2)
plot(nn45LmMdl.nn45Info.epoch, sqrt(nn45LmMdl.nn45Info.perf),'--','Color','#EDB120','LineWidth',2) %default color3
plot(nn45LmMdl.nn45Info.epoch, sqrt(nn45LmMdl.nn45Info.tperf),'*-','Color','#EDB120','LineWidth',2)
plot(ndt15LM.ndtInfo.epoch, sqrt(ndt15LM.ndtInfo.perf),'--','Color','#7E2F8E','LineWidth',2) 
plot(ndt15LM.ndtInfo.epoch, sqrt(ndt15LM.ndtInfo.tperf),'*-','Color','#7E2F8E','LineWidth',2)
plot(ndt10LM.ndtInfo.epoch, sqrt(ndt10LM.ndtInfo.perf),'--','Color','#77AC30','LineWidth',2) 
plot(ndt10LM.ndtInfo.epoch, sqrt(ndt10LM.ndtInfo.tperf),'*-','Color','#77AC30','LineWidth',2)
% legend('Tree,train','Tree,test','NDT,train','NDT,test','NN45,train','NN45,test')
legend('NDT LM train','NDT LM test','PINE, train','PINE, test'...
    ,'NDT15 LM train','NDT15 LM test','NDT10 LM train','NDT10 LM test')
xlabel('Epochs')
ylabel('RMSE')
title('Performance Comparison in RMSE')

sqrt(nn45LmMdl.nn45Info.perf(41))
sqrt(nn45LmMdl.nn45Info.tperf(41))

sqrt(ndtLmMld.ndtInfo.perf(41))
sqrt(ndtLmMld.ndtInfo.tperf(41))

sqrt(ndt15LM.ndtInfo.perf(41))
sqrt(ndt15LM.ndtInfo.tperf(41))

sqrt(ndt10LM.ndtInfo.perf(41))
sqrt(ndt10LM.ndtInfo.tperf(41))

nn45SGDMdl.nn45Info.TrainingRMSE(41*293)
nn45SGDMdl.nn45Info.ValidationRMSE(41*293)

ndtSGDMdl.ndtInfo.TrainingRMSE(41*293)
ndtSGDMdl.ndtInfo.ValidationRMSE(41*293)

ndtAdamMdl.ndtInfo.TrainingRMSE(41*293)
ndtAdamMdl.ndtInfo.ValidationRMSE(41*293)




