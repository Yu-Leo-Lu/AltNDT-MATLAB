function [P,train_idx, test_idx] = loadPINE()

File   = 'PINE_train_val_test_dataset_IrinaOptimal.mat';
P = load(File);  

test_idx = P.inds_test;
train_idx = P.inds_train{1,1};
train_idx = union(train_idx, P.inds_train{2,1});
train_idx = union(train_idx, P.inds_train{3,1});
train_idx = union(train_idx, P.inds_train{4,1});
train_idx = union(train_idx, P.inds_train{5,1});
end