startup

% load(fullfile(dir, 'results', 'nn45_200eps_lr1e-1_bs10000_mmt1e-1_statFeatPolarMLT'));
% 
% for iCell = 1:length(Stat)
%     Stat(iCell).LonRange = Stat(iCell).LonRange';
% end
% 
% 
% names = sprintf('nn45_200eps_lr1e-1_bs10000_mmt1e-1_scaledStatPolarMLT.mat');
% 
% 
% 
% 
% save(fullfile(dir, 'results', names), 'nn45', 'nn45Info',...
%     'procFcnsInput', 'settingsXTrain', 'Stat')
% 
% 
