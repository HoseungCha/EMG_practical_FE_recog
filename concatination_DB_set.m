%--------------------------------------------------------------------------
% Concatination of features from different DB set 
% train-less set + pratical set
%--------------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%--------------------------------------------------------------------------

%% get Feature set from DB(train-less experiement)
path_trainless ='E:\OneDrive_Hanyang\楷备\EMG_TrainLess_Expression\内靛\DB\ProcessedDB';
load(fullfile(path_trainless,'feat_set_combined_seg_60_using_ch4'));

%% get Feature set from DB(practical experiement)
path_practical = 'E:\OneDrive_Hanyang\楷备\EMG_FE_recognition_emotion\内靛\DB\ProcessedDB';
load(fullfile(path_practical,'feat_set'));
% get rid of expression(kiss)
Features(:,:,9,:,:) = [];

%% Concatination
feat_set_combined = cat(5,feat_set_combined,Features);

%% saving
save(fullfile(path_practical,'feat_set_combined_of_tless_prac_seg_60_using_ch4'),...
    'feat_set_combined');


