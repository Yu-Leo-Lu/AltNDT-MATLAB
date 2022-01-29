function [P,inds_train, inds_test] = loadPineTimeSeq()

startup

File   = 'PINE_train_val_test_dataset_IrinaOptimal.mat';
P = load(File);  

time = P.data_all.time;
[~, time_idx]= sort(time);
test_size = size(P.inds_test,1);

% test_size = 336k
% test indices is the largest 336k of all entries in time_idx
% train indices is the rest
inds_test = time_idx(time_idx>(max(time_idx)-test_size));
inds_train = time_idx(time_idx<=(max(time_idx)-test_size));

end