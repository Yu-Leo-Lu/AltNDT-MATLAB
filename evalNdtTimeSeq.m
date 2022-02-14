startup

% load sequential PINE data
[PINE,inds_train, inds_test] = loadPineTimeSeq();
trainIdx = inds_train{1};
testIdx = inds_test{1};

ndt25_40eps = load(fullfile(dir,'results','trainlm','TimeSeq','ndt25_40eps_timesep'));
nn45_40eps = load(fullfile(dir,'results','trainlm','TimeSeq', 'nn45_40eps_timesep'));

ndt25_40eps_Adam = load(fullfile(dir,'results','Adam','timeseq','ndtAdam_40eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1'));
nn45_40_eps_Adam = load(fullfile(dir,'results','Adam','timeseq', 'nn45_40eps_lr1e-2_bs10000_beta1_9e-1_beta2_9e-1'));

% RMSE plot time sequential
rmsePlot(ndt25_40eps, nn45_40eps, 'TimeSeq')
%---------------------- RMSE plot convergence SGD ----------------------
rmsePlotSGD(ndt25_40eps_Adam, nn45_40_eps_Adam, 293,1:40, 'Adam')
