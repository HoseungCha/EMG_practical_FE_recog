%----------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%---------------------------------------------------------------------

clc; clear all; close all;
parentdir=(fileparts(pwd));
addpath(genpath(fullfile(parentdir,'functions')));
%% 실험 정보

% Facial expression list
FE_name = {'Angry','Contemptuous','Disgust','Fear','Happy','Neutral','Sad','Surprised','Kiss'};
N_FaExp = length(FE_name);% Number of facial expression
N_Trl = 20; % Number of Trials

% trigger singals corresponding to each facial expression(emotion)
Trg_Inform = {"화남 ",1,1;"비웃음",1,2;"역겨움",1,3;"두려움",1,4;"행복",1,5;"무표정",1,6;"슬픔",1,7;"놀람",2,1;"키스",2,2};
FE_name = Trg_Inform(:,1);
Idx_trg = cell2mat(Trg_Inform(:,2:3));
clear Trg_Inform;

% filter parameters
SF2use = 2048;
fp.Fn = SF2use/2;
filter_order = 4; Fn = SF2use/2;
Notch_freq = [58 62];
BPF_cutoff_Freq = [20 450];
[nb,na] = butter(filter_order,Notch_freq/Fn,'stop');
[bb,ba] = butter(filter_order,BPF_cutoff_Freq/Fn,'bandpass');

% subplot 그림 꽉 차게 출력 관련 
make_it_tight = true; subplot = @(m,n,p) subtightplot (m, n, p, [0.01 0.05], [0.1 0.01], [0.1 0.01]);
if ~make_it_tight,  clear subplot;  end

% read file path of data
datapath = fullfile(parentdir,'DB','RawDB');
[Sname,Spath] = read_names_of_file_in_folder(datapath);
N_subject = length(Sname);

% experiments or feat extractions parameters
N_seg = 60;
N_feat = 28;
N_trl = 20;
i_comb = 1;
rc_matrix = [1,2;1,3;2,3]; %% 오른쪽 전극 조합
lc_matrix = [10,9;10,8;9,8]; %% 왼쪽 전극 조합

%% 결과 memory alloation
Features = zeros(N_seg,N_feat,N_FaExp,N_trl,N_subject);
% Features(:,:,event_s(i_emo,1),i_data,i_sub)
for i_sub= 1:N_subject
    
    sub_name = Sname{i_sub}(end-2:end);

    [fname,fpath] = read_names_of_file_in_folder(Spath{i_sub},'*bdf');
    
    % for saving feature Set (processed DB)
    time_set = cell(20,1);
    for i_data = 1:N_Trl
%     for i_data = 1
% fpath{i_data}
        
        try
            OUT = pop_biosig(fpath{i_data});
%             pop_biosig('E:\OneDrive_Hanyang\연구\EMG_FE_recognition_emotion\코드\DB\DB_cannot_be_used\006_이동재\12.bdf')
            %% load trigger when subject put on a look of facial expressoins
            %Trigger latency 및 FE 라벨
            temp = cell2mat(permute(struct2cell(OUT.event),[1 3 2]))';
%             time_set{i_data} = mat2cell(temp(:,2)/2048,ones(length(temp(:,2)),1),1);
            time_set{i_data} = temp(:,2)/2048;
        catch          
            time_set{i_data} = NaN;
        end
        
    end
    time_sets{i_sub} = padcat(time_set{:});

end
% time_set = time_set';
% padcat(time_set{:})
% 
% time_set(1,6:end) ={NaN};
% cell2mat(time_set(2:6,:));


