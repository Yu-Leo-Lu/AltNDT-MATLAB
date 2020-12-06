function [P,inds_train, inds_test] = loadPINE()

File   = 'PINE_train_val_test_dataset_IrinaOptimal.mat';
P = load(File);  

inds_test = P.inds_test;
inds_train = P.inds_train{1,1};
inds_train = union(inds_train, P.inds_train{2,1});
inds_train = union(inds_train, P.inds_train{3,1});
inds_train = union(inds_train, P.inds_train{4,1});
inds_train = union(inds_train, P.inds_train{5,1});
end