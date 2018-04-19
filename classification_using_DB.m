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
addpath(genpath(fullfile(fileparts(fileparts(fileparts(pwd))),'_matlab_toolbox')));
path_research=(fileparts(pwd));
addpath(genpath(fullfile(path_research,'functions')));

% Feature SET 가져오기
name_feat_file = 'feat_set_seg_30';
load(fullfile(path_research,'DB','DB_processed',name_feat_file));

%% DB set 가져오기
name_DB_file = 'feat_set_combined_seg_30_using_ch4';
load(fullfile(path_research,'DB','DB_processed_from_trainless',name_DB_file));
features_DB = feat_set_combined; clear feat_set_combined;
%% 실험 정보
[n_seg, n_feat, n_FE, n_trl, n_sub , n_emg_pair] = size(Features); % DB to be analyzed
n_sub_DB = size(features_DB,5); % Database
idx_sub = 1 : n_sub;
idx_trl = 1 : n_trl;
%% feature indexing
% when using DB of ch4 ver
idx_feat.RMS = 1:4;
idx_feat.WL = 5:8;
idx_feat.SampEN = 9:12;
idx_feat.CC = 13:28;
n_feat = 28;
%% feat names and indices
names_feat = fieldnames(idx_feat);
idx_feat = struct2cell(idx_feat);
n_ftype = length(names_feat);
%% decide how many number of tranfored feat from DB 
n_transforemd = 5;
% makeing folder for results 결과 저장 폴더 설정
% folder_name2make = ['T5_',name_feat_file]; % 폴더 이름
% path_made = make_path_n_retrun_the_path(fullfile(parentdir,...
%     'DB','dist'),folder_name2make); % 결과 저장 폴더 경로
%% memory allocation for reults
r_total = cell(n_emg_pair,1);
for i_emg_pair = 1 : n_emg_pair
r.acc = zeros(n_seg,n_trl,n_sub,n_transforemd+1);
r.output_n_target = cell(n_seg,n_trl,n_sub,n_transforemd+1);    
for i_sub = 1 : n_sub
    for i_trial = 1 : n_trl
        fprintf('i_sub:%d i_trial:%d\n',i_sub,i_trial);
        %% get similar feature from DB
        feat_t = cell(n_seg,n_FE);
        for i_seg = 1 : n_seg
            for i_FE = 1 : n_FE
                feat_t{i_seg,i_FE} = cell(1,n_ftype);
                for i_FeatName = 1 : n_ftype
                    
                    %% get DB with a specific feature
                    N_feat_interested = length(idx_feat{i_FeatName});
                    feat_ref = Features(i_seg,idx_feat{i_FeatName} ,i_FE,...
                        i_trial,i_sub,i_emg_pair)';
                    feat_DB = features_DB(:,idx_feat{i_FeatName} ,:,:,:);
%                     feat_ref = feat(i_seg,:,i_FE,i_trial,i_sub)';
                    feat_compr = feat_DB(i_seg,:,i_FE,:,:);
                    feat_compr = reshape(feat_compr,...
                        [N_feat_interested, n_trl*n_sub_DB]);
                    % just get 5 similar features
                    feat_t{i_seg,i_FE}{i_FeatName} = ...
                        dtw_search_n_transf(feat_ref, feat_compr, n_transforemd)';
                end
            end
        end
        %% arrange feat transformed and target
        % concatinating features with types
        feat_t = cellfun(@(x) cat(2,x{:}),feat_t,'UniformOutput',false);        
        % validate with number of transformed DB
        for n_t = 0: n_transforemd
            % get feature-transformed with number you want
            feat_trans = cellfun(@(x) x(1:n_t,:),feat_t,...
                'UniformOutput',false);
            % get size to have target
            size_temp = cell2mat(cellfun(@(x) size(x,1),...
                feat_trans(:,1),'UniformOutput',false));
            % feature transformed 
            feat_trans = cell2mat(feat_trans(:));
            % target for feature transformed 
            target_feat_trans = repmat(1:n_FE,sum(size_temp,1),1);
            target_feat_trans = target_feat_trans(:); 
            % feat for anlaysis
            feat_ref = reshape(permute(Features(:,:,:,i_trial,i_sub,i_emg_pair),...
                [1 3 2]),[n_seg*n_FE,n_feat]);
            target_feat_ref = repmat(1:n_FE,n_seg,1);
            target_feat_ref = target_feat_ref(:);
            % get input and targets for train DB
            input_train = cat(1,feat_ref,feat_trans);
            target_train = cat(1,target_feat_ref,target_feat_trans);
            % get input and targets for test DB
            input_test = reshape(permute(Features(:,: ,:,idx_trl~=i_trial,...
                i_sub,i_emg_pair),[1 4 3 2]),[n_seg*(n_trl-1)*n_FE,n_feat]);
            % train
            model.lda = fitcdiscr(input_train,target_train);
            % test
            output_test = predict(model.lda,input_test);
            % reshape ouput_test as <seg, trl, FE>
            output_test = reshape(output_test,[n_seg,(n_trl-1),n_FE]);
            output_mv_test = majority_vote(output_test);
            
            % reshape target test for acc caculation
            target_test = repmat(1:n_FE,(n_trl-1),1);
            target_test = target_test(:);
            for i_seg = 1 : n_seg
                ouput_seg = output_mv_test(i_seg,:)';
                r.acc(i_seg,i_trial,i_sub,n_t+1) = ...
                    sum(target_test==ouput_seg)/(n_FE*(n_trl-1))*100;
                r.output_n_target{i_seg,i_trial,i_sub,n_t+1} = ...
                    [ouput_seg,target_test];
            end
        end
    end
end
r_total{i_emg_pair} = r;
end

%% plot
for i_emg_pair = 1 : 3
    tmp = r_total{i_emg_pair};
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


