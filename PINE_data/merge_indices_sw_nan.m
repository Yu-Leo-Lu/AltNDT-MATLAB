load('PINE_train_val_test_dataset_sw_nan.mat')
load('PINE_train_val_test_dataset_indices_nan.mat')

%Then change names for each data_all and each feature_names

X1 = data_all_indices.X(:,1:6);
X2 = data_all_sw.X(:,1:4);
X3 = data_all_indices.X(:,7:end);
X4 = data_all_sw.X(:,5:end);
X = [X1,X2,X3,X4];
inds_nan = find(any(isnan(X), 2));

fn1 = feature_names_indices(1:6);
fn2 = feature_names_sw(1:4);
fn3 = feature_names_indices(7:end);
fn4 = feature_names_sw(5:end);
feature_names = [fn1; fn2; fn3; fn4];

t = data_all_indices.t;
time = data_all_indices.time;

data_all = struct('X', X, 't', t, 'time', time);