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
parentdir=(fileparts(pwd));
addpath(genpath(fullfile(parentdir,'functions')));
%% 실험 정보

% Facial expression list
FE_name = {'Angry','Contemptuous','Disgust','Fear','Happy','Neutral','Sad','Surprised','Kiss'};
N_FaExp = length(FE_name);% Number of facial expression
N_Trl = 20; % Number of Trials
datapath = fullfile(parentdir,'DB','RawDB');

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
    count_i_data = 0;
    for i_data = 1:N_Trl
%     for i_data = 1
        count_i_data = count_i_data + 1;
        OUT = pop_biosig(fpath{i_data});
        
       %% load trigger when subject put on a look of facial expressoins
        %Trigger latency 및 FE 라벨
        temp = cell2mat(permute(struct2cell(OUT.event),[1 3 2]))';
        temp(:,1) = temp(:,1)./128;
        Idx_trg_obtained = reshape(temp(:,1),[2,size(temp,1)/2])';
        temp = reshape(temp(:,2),[2,size(temp,1)/2])';
        lat_trg = temp(:,1);
        [~,idx_in_order] = sortrows(Idx_trg_obtained);
        % [idx_in_order,(1:N_FaExp)'], idices in order corresponding to
        % emotion label
        temp = sortrows([idx_in_order,(1:length(idx_in_order))'],1); 
        FE_sequence = temp(:,2); clear Idx_trg_obtained;
        
        % for data consistency, put in fake data
        if size(FE_sequence,1) == 8
            FE_sequence = [FE_sequence;9];
            lat_trg = [lat_trg;0];
        end
        
        %% get raw data and bipolar configuration
        raw_data = double(OUT.data');
              
         % channel configuration
         
        temp_chan = cell(1,6);
        % get raw data and bipolar configuration        
        emg_bip.RZ= OUT.data(rc_matrix(i_comb,1),:) - OUT.data(rc_matrix(i_comb,2),:);%Right_Zygomaticus
        emg_bip.RF= OUT.data(4,:) - OUT.data(5,:); %Right_Frontalis
        emg_bip.LF= OUT.data(6,:) - OUT.data(7,:); %Left_Corrugator
        emg_bip.LZ= OUT.data(lc_matrix(i_comb,1),:) - OUT.data(lc_matrix(i_comb,2),:); %Right_Zygomaticus
        
        bp_data = double(cell2mat(struct2cell(emg_bip)))';


        %% Filtering
        filtered_data = filter(nb, na, bp_data,[],1);
        filtered_data = filter(bb, ba, filtered_data, [],1);
        
        % for plot
%         figure;plot(filtered_data)

        %% Feat extration with windows 
        % increase size = 0.05;, win size = 0.1;
        winsize = floor(0.1*SF2use); wininc = floor(0.05*SF2use); 
        % 0.1초 윈도우, 0.05초 씩 증가
        N_window = floor((length(filtered_data) - winsize)/wininc)+1;
        temp_feat = zeros(N_window,N_feat); Window_Endsamples = zeros(N_window,1);
        st = 1;
        en = winsize;
        for i = 1: N_window
            Window_Endsamples(i) = en;
            curr_win = filtered_data(st:en,:);
            temp_rms = sqrt(mean(curr_win.^2));
            temp_CC = featCC(curr_win,4);
            temp_WL = sum(abs(diff(curr_win,2)));
            temp_SampEN = SamplEN(curr_win,2);
            temp_feat(i,:) = [temp_CC,temp_rms,temp_SampEN,temp_WL];
            % moving widnow
            st = st + wininc;
            en = en + wininc;                 
        end

        %% cutting trigger 
        idx_TRG_Start = zeros(N_FaExp,1);
        for i_emo_orer_in_this_exp = 1 : N_FaExp
            idx_TRG_Start(i_emo_orer_in_this_exp,1) = find(Window_Endsamples >= lat_trg(i_emo_orer_in_this_exp),1);
        end
        
        %% To confirm the informaion of trrigers were collected right
        hf =figure(i_sub);
        hf.Position = [-2585 -1114 1920 1091];
        subplot(N_Trl,1,i_data);
        plot(temp_feat(:,17:20));
        hold on;
        stem(idx_TRG_Start,repmat(100,[N_FaExp,1]));
        ylim([1 300]);
%         subplot(N_Trl,2,2*i_data);
%         plot(temp_feat(:,1:6));
%         hold on;
%         stem(idx_TRG_Start,ones([N_FE,1]));
%         drawnow;
        
       %% Get Feature sets(preprocessed DB)
       
        for i_emo_orer_in_this_exp = 1 : N_FaExp
            Features(:,:,FE_sequence(i_emo_orer_in_this_exp),count_i_data,i_sub) = ...
                        temp_feat(idx_TRG_Start(i_emo_orer_in_this_exp):...
                        idx_TRG_Start(i_emo_orer_in_this_exp)+floor((3*SF2use)/wininc)-1 ,:); 
        end 
    end  
    %% plot the DB 
    c = getframe(hf);
    imwrite(c.cdata,fullfile(parentdir,'DB','DB_inspection',...
        [sub_name(1:3),'.jpg']));
    close(hf);
end
%% 결과 저장
save(fullfile(parentdir,'DB','ProcessedDB',sprintf('feat_set')),...
    'Features');



