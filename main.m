%----------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%---------------------------------------------------------------------

clc; clear all; close all;
addpath(genpath(fullfile(cd,'functions')));

% subplot 그림 꽉 차게 출력 관련 
make_it_tight = true; subplot = @(m,n,p) subtightplot (m, n, p, [0.01 0.05], [0.1 0.01], [0.1 0.01]);
if ~make_it_tight,  clear subplot;  end

% 실험 정보
% 트리거 정보
Trg_Inform = {"화남 ",1,1;"비웃음",1,2;"역겨움",1,3;"두려움",1,4;"행복",1,5;"무표정",1,6;"슬픔",1,7;"놀람",2,1;"키스",2,2};
FE_name = Trg_Inform(:,1);
Idx_trg = cell2mat(Trg_Inform(:,2:3));
clear Trg_Inform;

N_Trl = 20; %trial
N_FE = length(FE_name);  %facial expression

% read file path of data
[Sname,Spath] = read_names_of_file_in_folder(fullfile(cd,'DB'));
N_subject = length(Sname);

% 파라미터
SF2use = 2048;
fp.Fn = SF2use/2;
filter_order = 4; Fn = SF2use/2;
Notch_freq = [58 62];
BPF_cutoff_Freq = [20 450];
[nb,na] = butter(filter_order,Notch_freq/Fn,'stop');
[bb,ba] = butter(filter_order,BPF_cutoff_Freq/Fn,'bandpass');

i_comb = 1;
rc_matrix = [1,2;1,3;2,3]; %% 오른쪽 전극 조합
lc_matrix = [10,9;10,8;9,8]; %% 왼쪽 전극 조합

%% 결과 memory alloation
Features = zeros(30,28,N_FE,N_Trl,N_subject);
% Features(:,:,event_s(i_emo,1),i_data,i_sub)
for i_sub= 1:N_subject
    
    sub_name = Sname{i_sub}
%     FE_list_DB_sub = FE_list_DB(:,i_sub);
%     N_Trl = length(find(~isnan(FE_list_DB_sub)));
%     FE_list_DB_sub = FE_list_DB_sub(1:N_Trl);
%     N_Trl  = N_Trl/N_FE;
%     FE_list_DB_sub = reshape(FE_list_DB_sub,[N_FE,N_Trl]);
%     count_i_data = 0;
    for i_trl = 1 : N_Trl
%         count_i_data = count_i_data + 1;
        OUT = pop_biosig(fullfile(cd,'DB',sub_name,sprintf('%d.bdf',i_trl)));
        
        % Trigger latency 및 FE 라벨
        temp = cell2mat(permute(struct2cell(OUT.event),[1 3 2]))';
        temp(:,1) = temp(:,1)./128;
        Idx_trg_obtained = reshape(temp(:,1),[2,9])';
        temp = reshape(temp(:,2),[2,9])';
        lat_trg = temp(:,1);
        [~,temp] = sortrows(Idx_trg_obtained);
        temp = sortrows([temp,(1:9)'],1);
        label_FE = temp(:,2); clear Idx_trg_obtained;

        % get raw data and bipolar configuration        
        emg_bip.RZ= OUT.data(rc_matrix(i_comb,1),:) - OUT.data(rc_matrix(i_comb,2),:);
        emg_bip.RF= OUT.data(4,:) - OUT.data(5,:);
        emg_bip.LF= OUT.data(6,:) - OUT.data(7,:);
        emg_bip.LZ= OUT.data(lc_matrix(i_comb,1),:) - OUT.data(lc_matrix(i_comb,2),:);
            
         % channel configuration
        temp = double(cell2mat(struct2cell(emg_bip)))';

        % Filtering
        temp = filter(nb, na, temp,[],1);
        temp = filter(bb, ba, temp, [],1);

        % Feat extration
        winsize = floor(0.1*SF2use); wininc = floor(0.1*SF2use); 
        % 0.1초 윈도우, 0.1초 씩 증가
        N_window = floor((length(temp) - winsize)/wininc)+1;
        temp_feat = zeros(N_window,28); Window_Endsamples = zeros(N_window,1);
        st = 1;
        en = winsize;
        for i = 1: N_window
%             if (i_ch==1)
            Window_Endsamples(i) = en;
%             end
            curr_win = temp(st:en,:);

            temp_rms = sqrt(mean(curr_win.^2));
            temp_CC = featCC(curr_win,4);
            temp_WL = sum(abs(diff(curr_win,2)));
            for i_ch = 1 : 4
               r = 0.2*std(curr_win(:,i_ch),1); %% Standard deviation 
               temp_SampEN(1,i_ch) = FastSampEn(curr_win(:,i_ch), 2, r, 1); %SampEn
            end
%             temp_SampEN = SamplEN(curr_win,2);
            temp_feat(i,:) = [temp_rms,temp_WL,temp_SampEN,temp_CC];
            % moving widnow
            st = st + wininc;
            en = en + wininc;                 
        end

        % cutting trigger 
        idx_TRG_Start = zeros(N_FE,1);
        for i_emo = 1 : N_FE
            idx_TRG_Start(i_emo,1) = find(Window_Endsamples >= lat_trg(label_FE==i_emo),1);
        end
        
        % To confirm the informaion of trrigers were collected right
%         hf =figure(i_sub);
%         plot(temp_feat(:,4));
%         hold on;
%         stem(idx_TRG_Start,repmat(100,[N_FE,1]));
%         ylim([1 150]);
        drawnow;
        
       %% Get fTDDfeats of interested segments (facial expression task)
        for i_emo = 1 : N_FE
            Features(:,:,i_emo,i_trl,i_sub) = ...
                        temp_feat(idx_TRG_Start(i_emo):idx_TRG_Start(i_emo)+floor((3*SF2use)/wininc)-1 ,:); 
        end 
    end  
end
%% 결과 저장
% % formatOut = 'yy_mm_dd_HHMMSS_';
% % current_time=datestr(now,formatOut);
% % 
% % save(fullfile(cd,'result',...
% %     ['Features_extracted',current_time]),'Features');

% save(fullfile(cd,'result',sprintf('FEATS_2ND.mat')),'Features');
 
 %
% %%%%%% plot        
% f_sz = ceil(length(filtered_data)/2);
% f = SF2use/2*linspace(0,1,f_sz);
% f_X=fft(tem_filtered_data); 
% f_y=fft(filtered_data); 
% 
% figure();
% subplot(2,1,1);
% stem(f, abs(f_X(1:f_sz)));
% 
% subplot(2,1,2);
% stem(f, abs(f_y(1:f_sz)))









