function [P,inds_train, inds_test] = loadPineTimeSeq()

startup

File   = 'PINE_train_val_test_dataset_IrinaOptimal.mat';
P = load(File);  

time = P.data_all.time;
[~, time_idx]= sort(time);
test_size = size(P.inds_test,1);
all_size = size(P.data_all.t,1);
% test_size = 336k
% test indices is the largest 336k of all entries in time_idx
% train indices is the rest

inds_train = cell(5,1);
inds_test = cell(5,1);

% ts: test_size, a: all_size
% shift training and testing to previous timeline
% so that test size is always constant
% |   a-ts   | ts |
% |a-2ts| ts | ts |
% ...
for i = 1:5
    inds_train{i} = time_idx(1:all_size-i*test_size);
    inds_test{i} = time_idx(all_size-i*test_size+1:all_size-(i-1)*test_size);
end

end