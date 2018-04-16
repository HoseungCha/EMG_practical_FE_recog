%--------------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
% 2017.09.13 DTW변환 100개 늘림(함수로 간략화시킴)
%--------------------------------------------------------------------------
clc; close all; clear all;
% get tool box
addpath(genpath(fullfile('E:\Hanyang\연구','_matlab_toolbox')));
parentdir=(fileparts(pwd));
addpath(genpath(fullfile(parentdir,'functions')));

%% Feature SET 가져오기
name_feat_file = 'feat_set_seg_30';
load(fullfile(parentdir,'DB','ProcessedDB',name_feat_file));

%% DB set 가져오기
name_DB_file = 'feat_set_combined_seg_30_using_ch4';
load(fullfile(parentdir,'DB','DB_processed_from_trainless',name_DB_file));
Features_DB = feat_set_combined; clear feat_set_combined;
%% 실험 정보
[N_Seg, N_Feat, N_FE, N_trial, N_sub , N_emgpair] = size(Features); % DB to be analyzed
N_sub_DB = size(Features_DB,5); % Database
idx_sub = 1 : N_sub;
idx_trl = 1 : N_trial;
%% feature indexing
% when using DB of ch4 ver
idx_feat.RMS = 1:4;
idx_feat.WL = 5:8;
idx_feat.SampEN = 9:12;
idx_feat.CC = 13:28;
N_feat = 28;
%% feat names and indices
names_feat = fieldnames(idx_feat);
idx_feat = struct2cell(idx_feat);
N_ftype = length(names_feat);
%% decide how many number of tranfored feat from DB 
N_transforemd = 5;
N_emg_pair = 3;
% makeing folder for results 결과 저장 폴더 설정
% folder_name2make = ['T5_',name_feat_file]; % 폴더 이름
% path_made = make_path_n_retrun_the_path(fullfile(parentdir,...
%     'DB','dist'),folder_name2make); % 결과 저장 폴더 경로
%% memory allocation for reults
R_Total = cell(N_emg_pair,1);
for i_emg_pair = 1 : N_emg_pair
R.acc = zeros(N_Seg,N_trial,N_sub,N_transforemd+1);
R.output_n_target = cell(N_Seg,N_trial,N_sub,N_transforemd+1);    
for i_sub = 1 : N_sub
    for i_trial = 1 : N_trial
        fprintf('i_sub:%d i_trial:%d\n',i_sub,i_trial);
        %% get similar feature from DB
        T = cell(N_Seg,N_FE);
        for i_seg = 1 : N_Seg
            for i_FE = 1 : N_FE
                T{i_seg,i_FE} = cell(1,N_ftype);
                for i_FeatName = 1 : N_ftype
                    
                    %% get DB with a specific feature
                    N_feat_interested = length(idx_feat{i_FeatName});
                    feat_ref = Features(i_seg,idx_feat{i_FeatName} ,i_FE,...
                        i_trial,i_sub,i_emg_pair)';
                    feat_DB = Features_DB(:,idx_feat{i_FeatName} ,:,:,:);
%                     feat_ref = feat(i_seg,:,i_FE,i_trial,i_sub)';
                    feat_compr = feat_DB(i_seg,:,i_FE,:,:);
                    feat_compr = reshape(feat_compr,...
                        [N_feat_interested, N_trial*N_sub_DB]);
                    % just get 5 similar features
                    T{i_seg,i_FE}{i_FeatName} = ...
                        dtw_search_n_transf(feat_ref, feat_compr, N_transforemd)';
                end
            end
        end
        %% arrange feat transformed and target
        % concatinating features with types
        T = cellfun(@(x) cat(2,x{:}),T,'UniformOutput',false);        
        % validate with number of transformed DB
        for N_trans = 0: N_transforemd
            % get feature-transformed with number you want
            feat_trans = cellfun(@(x) x(1:N_trans,:),T,...
                'UniformOutput',false);
            % get size to have target
            size_temp = cell2mat(cellfun(@(x) size(x,1),...
                feat_trans(:,1),'UniformOutput',false));
            % feature transformed 
            feat_trans = cell2mat(feat_trans(:));
            % target for feature transformed 
            target_feat_trans = repmat(1:N_FE,sum(size_temp,1),1);
            target_feat_trans = target_feat_trans(:); 
            % feat for anlaysis
            feat_ref = reshape(permute(Features(:,:,:,i_trial,i_sub,i_emg_pair),...
                [1 3 2]),[N_Seg*N_FE,N_feat]);
            target_feat_ref = repmat(1:N_FE,N_Seg,1);
            target_feat_ref = target_feat_ref(:);
            % get input and targets for train DB
            input_train = cat(1,feat_ref,feat_trans);
            target_train = cat(1,target_feat_ref,target_feat_trans);
            % get input and targets for test DB
            input_test = reshape(permute(Features(:,: ,:,idx_trl~=i_trial,...
                i_sub,i_emg_pair),[1 4 3 2]),[N_Seg*(N_trial-1)*N_FE,N_feat]);
            % train
            model.lda = fitcdiscr(input_train,target_train);
            % test
            output_test = predict(model.lda,input_test);
            % reshape ouput_test as <seg, trl, FE>
            output_test = reshape(output_test,[N_Seg,(N_trial-1),N_FE]);
            output_mv_test = majority_vote(output_test);
            
            % reshape target test for acc caculation
            target_test = repmat(1:N_FE,(N_trial-1),1);
            target_test = target_test(:);
            for i_seg = 1 : N_Seg
                ouput_seg = output_mv_test(i_seg,:)';
                R.acc(i_seg,i_trial,i_sub,N_trans+1) = ...
                    sum(target_test==ouput_seg)/(N_FE*(N_trial-1))*100;
                R.output_n_target{i_seg,i_trial,i_sub,N_trans+1} = ...
                    [ouput_seg,target_test];
            end
        end
    end
end
R_Total{i_emg_pair} = R;
end

%% plot
for i_emg_pair = 1 : 3
    tmp = R_Total{i_emg_pair};
    tmp = permute(mean(mean(tmp.acc,2),3),[1 4 2 3]);
    plot(tmp)
end
% save at directory of DB\dist
%             save(fullfile(path_made,['T_',num2str(i_sub),'_',...
%                 num2str(i_trial),'_',names_feat{i_FeatName},'_5.mat']),'T');

% function [xt] = dtw_search_n_transf(x1, x2, N_s)
% % parameters
% window_width = 3;
% max_slope_length = 2;
% speedup_mode = 1;
% DTW_opt.nDivision_4Resampling = 10;
% DTW_opt.max_slope_length = 3;
% DTW_opt.speedup_mode = 1;
% 
% [N_f, N]= size(x2); dist = zeros(N,1);
% for i = 1 : N
%     dist(i) = fastDTW(x1, x2(:,i),max_slope_length, ...
%         speedup_mode, window_width );
% end
% % Sort
% [~, sorted_idx] = sort(dist);
% % xs= x2(:,sorted_idx(1:N_s));
% xt = zeros(N_f,N_s);
% for i = 1 : N_s
%     xt(:,i)= transfromData_accRef_usingDTW(x2(:,sorted_idx(i)), x1, DTW_opt);
% end
% end


