startup
addpath('prepareData')

[PINE,trainIdx, testIdx] = loadPINE();

omni2001 = load(fullfile(dir, 'PINE_data', 'indices2001', 'OmniData_2001'));
ASYM2001 = load(fullfile(dir, 'PINE_data', 'indices2001', 'Minute_ASYM_2001'));
AE2001 = load(fullfile(dir, 'PINE_data', 'indices2001', 'Minute_AE_2001'));

% reshape AE: go along column, so need to permute first
%             (1:60,1,1), (1:60,2,1), (1:60,3,1),...,
%             (1:60,1,2), (1:60,2,2), (1:60,3,2),...,
%             (1:60,1,3), (1:60,2,3), (1:60,3,3),...,

AE = AE2001.DataRecord.AE.Value(1:365,:,:); % 365*24*60
AE = permute(AE,[3,2,1]); % 60*24*365
AE = reshape(AE,[numel(AE),1]); 
% (:,:,2) is H component of Sym 365x24
SymH = ASYM2001.DataRecord.SYM.Value(:,:,:,2);
SymH = permute(SymH,[3,2,1]);
SymH = reshape(SymH, [numel(SymH),1]);

kp = omni2001.DataRecord.DataField(11).Value;
kp = kp./10; % unit scale to match Irina's dataset
F107 = omni2001.DataRecord.DataField(15).Value;

% use hourly mean to generate constant minute mean
% repeat rows(feature col transpose)by 60 times
% reshape to by column
temp = repmat(kp',60,1);
kp = temp(:);
temp = repmat(F107',60,1);
F107 = temp(:);

% let p store array of past hours
% for each feature, say AE in hrs, compute rolling mean in previous m hrs
% movmean with history of m hours, m even:
% m/2 integer, omit first m/2 elements and last m/2-1 elements

p = [3,6,12,24,36,48]*60; % hours to minutes 
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

figure
plot(kp(1:500))
xlabel('Minute')
ylabel('kp')
title('kp')

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
title('F10.7 in 2001 data from Tokyo');
xlim([70,300])
subplot(2,1,2)
histogram(PINE.data_all.X(:,4))
title('F10.7 from 2012 to 2016 from Irina');
xlim([70,300])

figure
plot(F107(1:500))
xlabel('Minute')
ylabel('F107')
title('F107')

% check AE from omni vs minutely
AEOmni = omni2001.DataRecord.DataField(16).Value;
AEminute = AE2001.DataRecord.AE.Value(1:365,:,:); % 365*24*60
AEhour = NaN(size(AEminute,1), size(AEminute,2));
for i = 1:size(AEminute,1)
    for j = 1:size(AEminute,2)
        AEhour(i,j) = nanmean(squeeze(AEminute(i,j,:)));
    end
end
AEhour = reshape(AEhour', [numel(AEhour),1]);

figure
subplot(3,1,1)
histogram(AEhour)
xlim([-200,3000])
title('AE from minute avg');
subplot(3,1,2)
histogram(AEOmni)
xlim([-200,3000])
title('AE from omni');
subplot(3,1,3)
histogram(PINE.data_all.X(:,1))
xlim([-200,3000])
title('AE from Irina');

figure
mins = 240;
tiles = tiledlayout(2,1);
% Plot in tiles
nexttile, plot(AEOmni(1:mins)), title('AE from Omni')
nexttile, plot(AEhour(1:mins)), title('AE from minute avg')
% Specify common title, X and Y labels
title(tiles, ['first ', num2str(mins),' hours of AE'])
xlabel(tiles, 'in Hours')
ylabel(tiles, 'AE Value')

% end of debug



