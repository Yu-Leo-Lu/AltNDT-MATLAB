startup
[PINE,trainIdx, testIdx] = loadPINE();

omni2001 = load(fullfile(dir, 'PINE_data', 'indices2001', 'OmniData_2001'));
ASYM2001 = load(fullfile(dir, 'PINE_data', 'indices2001', 'Minute_ASYM_2001'));


AE = omni2001.DataRecord.DataField(16).Value;
kp = omni2001.DataRecord.DataField(11).Value;
kp = kp./10; % unit scale to match Irina's dataset
F107 = omni2001.DataRecord.DataField(15).Value;
SymH = ASYM2001.DataRecord.SYM.HourlyMean(:,:,2); % 365x24
% reshape go along with rows, so use transpose of matrix SymH
SymH = reshape(SymH', [size(SymH,1)*size(SymH,2),1]); % 8760x1

% let p store array of past hours
% for each feature, say AE in hrs, compute rolling mean in previous m hrs
% movmean with history of m hours, m even:
% m/2 integer, omit first m/2 elements and last m/2-1 elements

p = [3,6,12,24,36,48];
AERolling = getHistory(p, AE);
kpRolling = getHistory(p, kp);
SymHRolling = getHistory(p, SymH);
F107Rolling = getHistory(p, F107);

% generate data in 2001 as Irina's format, without L and MLT
data2001 = [AE,kp,SymH,F107,NaN(length(AE),1), NaN(length(AE),1),...,
    AERolling, kpRolling, SymHRolling, F107Rolling];

names = sprintf('data2001');
save(fullfile(dir, 'PINE_data', names), 'data2001')

% check AE
figure
subplot(2,1,1)
histogram(AE)
xlim([-200,3000])
subplot(2,1,2)
histogram(PINE.data_all.X(:,1))
xlim([-200,3000])

% check kp
% kp is multiplied by 10 in data2001? so scaled by 1/10 when load
figure
subplot(2,1,1)
histogram(kp)
title('kp in 2001 data from Tokyo');
xlim([-1,10])
subplot(2,1,2)
histogram(PINE.data_all.X(:,2))
title('kp from 2012 to 2016 from Irina');
xlim([-1,10])

% check SymH
figure
subplot(2,1,1)
histogram(SymH)
xlim([-400,100])
subplot(2,1,2)
histogram(PINE.data_all.X(:,3))
xlim([-400,100])

% check F107
figure
subplot(2,1,1)
histogram(F107)
xlim([70,300])
subplot(2,1,2)
histogram(PINE.data_all.X(:,4))
xlim([70,300])



