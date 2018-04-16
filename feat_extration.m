%--------------------------------------------------------------------------
% feat extracion code for practical facial expression
%--------------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%--------------------------------------------------------------------------
clc; clear all; close all;
% get tool box
addpath(genpath(fullfile('E:\Hanyang\연구','_matlab_toolbox')));
path_parent=(fileparts(pwd));
addpath(genpath(fullfile(path_parent,'functions')));
%% 실험 정보
% Facial expression list
Name_FE = {'Angry','Contemptuous','Disgust','Fear','Happy','Neutral','Sad','Surprised','Kiss'};
N_FaExp = length(Name_FE);% Number of facial expression
N_Trl = 20; % Number of Trials
path_rawDB = fullfile(path_parent,'DB','RawDB');
% trigger singals corresponding to each facial expression(emotion)
Name_Trg = {"화남 ",1,1;"비웃음",1,2;"역겨움",1,3;"두려움",1,4;"행복",1,5;"무표정",1,6;"슬픔",1,7;"놀람",2,1;"키스",2,2};
Name_FE = Name_Trg(:,1);
Idx_trg = cell2mat(Name_Trg(:,2:3));
clear Name_Trg;

%% filter parameters
fp.SF2use = 2048;
fp.filter_order = 4; fp.Fn = fp.SF2use/2;
fp.Notch_freq = [58 62];
fp.BPF_cutoff_Freq = [20 450];
[fp.nb,fp.na] = butter(fp.filter_order,fp.Notch_freq/fp.Fn,'stop');
[fp.bb,fp.ba] = butter(fp.filter_order,fp.BPF_cutoff_Freq/fp.Fn,'bandpass');

% subplot 그림 꽉 차게 출력 관련 
make_it_tight = true; subplot = @(m,n,p) subtightplot (m, n, p, [0.01 0.05], [0.1 0.01], [0.1 0.01]);
if ~make_it_tight,  clear subplot;  end

%% read file path of data
[Name_subject,path_subjects] = read_names_of_file_in_folder(path_rawDB);
N_subject = length(Name_subject);
%% decide number of segments in 3-sec long EMG data
N_seg = 30; % choose 30 or 60
%% experiments or feat extractions parameters
N_feat = 28;
N_trl = 20;
N_comb = 3;
N_ch = 4;
rc_matrix = [1,2;1,3;2,3]; %% 오른쪽 전극 조합
lc_matrix = [10,9;10,8;9,8]; %% 왼쪽 전극 조합
Time_expression = 3; % 3-sec
Time_window = 0.1;
wininc = floor((Time_expression/N_seg)*fp.SF2use); 
%% 결과 memory alloation
Features = zeros(N_seg,N_feat,N_FaExp,N_trl,N_subject,N_comb);
% Features(:,:,event_s(i_emo,1),i_data,i_sub)
for i_comb = 1 : N_comb
for i_sub= 1:N_subject
    
    sub_name = Name_subject{i_sub}(end-2:end); %subject name
    % read BDF
    [~,path_file] = read_names_of_file_in_folder(path_subjects{i_sub},'*bdf');
   
    % for saving feature Set (processed DB)
    count_i_data = 0;
    for i_trl = 1:N_Trl
%     for i_data = 1
        count_i_data = count_i_data + 1;
        OUT = pop_biosig(path_file{i_trl});
        
       %% load trigger when subject put on a look of facial expressoins
        %Trigger latency 및 FE 라벨
        temp = cell2mat(permute(struct2cell(OUT.event),[1 3 2]))';
        temp(:,1) = temp(:,1)./128;
        if i_sub==13 && i_trl == 18 %% exception for a data
            % trigger may be superpositioned
            temp(16,:) = temp(15,:);temp(15,1) = 1;temp(16,1) = 6;
        end
        Idx_trg_obtained = reshape(temp(:,1),[2,size(temp,1)/2])';
        temp = reshape(temp(:,2),[2,size(temp,1)/2])';
        lat_trg = temp(:,1);
        [~,idx_in_order] = sortrows(Idx_trg_obtained);
        % [idx_in_order,(1:N_FaExp)'], idices in order corresponding to
        % emotion label
        temp = sortrows([idx_in_order,(1:length(idx_in_order))'],1); 
        idx_seq_FE = temp(:,2); clear Idx_trg_obtained temp;
        
        % for data consistency, put in fake data
        if size(idx_seq_FE,1) == 8
            idx_seq_FE = [idx_seq_FE;9];
            lat_trg = [lat_trg;0];
        end
        
        %% get raw data and bipolar configuration
        raw_data = double(OUT.data'); % raw data
        temp_chan = cell(1,6);
        % get raw data and bipolar configuration        
        emg_bip.RZ= OUT.data(rc_matrix(i_comb,1),:) - OUT.data(rc_matrix(i_comb,2),:);%Right_Zygomaticus
        emg_bip.RF= OUT.data(4,:) - OUT.data(5,:); %Right_Frontalis
        emg_bip.LF= OUT.data(6,:) - OUT.data(7,:); %Left_Corrugator
        emg_bip.LZ= OUT.data(lc_matrix(i_comb,1),:) - OUT.data(lc_matrix(i_comb,2),:); %Right_Zygomaticus
        bp_data = double(cell2mat(struct2cell(emg_bip)))';
        clear emg_bip out;
        %% Filtering
        filtered_data = filter(fp.nb, fp.na, bp_data,[],1);
        filtered_data = filter(fp.bb, fp.ba, filtered_data, [],1);
        clear bp_data;
        % for plot
%         figure;plot(filtered_data)
        %% Feat extration with windows 
        winsize = floor(Time_window*fp.SF2use); % win
%         wininc = floor(0.05*SF2use); 
        N_window = floor((length(filtered_data) - winsize)/wininc)+1;
        temp_feat = zeros(N_window,N_feat); idx_trg_as_window = zeros(N_window,1);
        st = 1;
        en = winsize;
        for i = 1: N_window
            idx_trg_as_window(i) = en;
            curr_win = filtered_data(st:en,:);
            temp_rms = sqrt(mean(curr_win.^2));
            temp_CC = featCC(curr_win,N_ch);
            temp_WL = sum(abs(diff(curr_win,2)));
            temp_SampEN = SamplEN(curr_win,2);
%             temp_feat(i,:) = [temp_CC,temp_rms,temp_SampEN,temp_WL];
            temp_feat(i,:) = [temp_rms,temp_WL,temp_SampEN,temp_CC];
            % moving widnow
            st = st + wininc;
            en = en + wininc;                 
        end
        clear temp_rms temp_CC temp_WL temp_SampEN st en
 
        %% cutting trigger 
        idx_TRG_Start = zeros(N_FaExp,1);
        for i_emo_orer_in_this_exp = 1 : N_FaExp
            idx_TRG_Start(i_emo_orer_in_this_exp,1) = find(idx_trg_as_window >= lat_trg(i_emo_orer_in_this_exp),1);
        end
        
        %% To confirm the informaion of trrigers were collected right
%         hf =figure(i_sub);
%         hf.Position = [-2585 -1114 1920 1091];
%         subplot(N_Trl,1,i_data);
%         plot(temp_feat(:,17:20));
%         hold on;
%         stem(idx_TRG_Start,repmat(100,[N_FaExp,1]));
%         ylim([1 300]);
%         subplot(N_Trl,2,2*i_data);
%         plot(temp_feat(:,1:6));
%         hold on;
%         stem(idx_TRG_Start,ones([N_FE,1]));
%         drawnow;
        
       %% Get Feature sets(preprocessed DB)
        for i_emo_orer_in_this_exp = 1 : N_FaExp
            Features(:,:,idx_seq_FE(i_emo_orer_in_this_exp),count_i_data,i_sub,i_comb) = ...
                        temp_feat(idx_TRG_Start(i_emo_orer_in_this_exp):...
                        idx_TRG_Start(i_emo_orer_in_this_exp)+floor((Time_expression*fp.SF2use)/wininc)-1 ,:); 
        end 
    end  
    %% plot the DB 
%     c = getframe(hf);
%     imwrite(c.cdata,fullfile(parentdir,'DB','DB_inspection',...
%         [sub_name(1:3),'.jpg']));
%     close(hf);
end
end
%% get rid of kiss expression
Features(:,:,9,:,:,:) = [];
%% 결과 저장
save(fullfile(path_parent,'DB','ProcessedDB',sprintf('feat_set_seg_%d',...
    N_seg)),'Features');

% save(fullfile(path_parent,'DB','ProcessedDB',sprintf('feat_set')),...
%     'Features');



