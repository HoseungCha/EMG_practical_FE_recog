%-------------------------------------------------------------------------%
% 1. feat_extraction.m
% 2. classficiation_using_DB.m  %---current code---%
%-------------------------------------------------------------------------%
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%-------------------------------------------------------------------------%
clc; close all; clear all;

%-----------------------Code anlaysis parmaters----------------------------
% load feature set, which was extracted by feat_extration.m
name_feat_file = 'feat_set_seg_30';

% load feature set, from different DB, which was extracted by
% feat_extraion.m in git_EMG_train_less_FE_recog
name_DB_file = 'feat_set_combined_seg_30_using_ch4';

% decide if you applied train-less algorithm using DB, or DB of other expreiment
id_DBtype = 'DB_both'; % 'DB_own' , 'DB_other' , 'DB_both'

% decide expression to classify
% ["화남(1)";"비웃음(2)";"역겨움(3)";"두려움(4)";"행복(5)";"무표정(6)";"슬픔(7)";
% "놀람(8)"];
% idx_FE2classfy =[1,5,7,8];% 화남, 놀람, 행복(기쁨), 슬픔 <교수님 선정>
idx_FE2classfy =[1,3,5,8];% 화남, 놀람, 행복(기쁨), 역겨움 <교수님 선정>

% decide number of tranfored feat from DB 
n_transforemd = 0;

% decide which attibute to be compared when applying train-less algoritm
% [n_seg:30, n_feat:28, n_FE:8, n_trl:20, n_sub:30]
% 'all' : [:,:,:,:,:], 'Only_Seg' : [i_seg,:,:,:,:], 'Seg_FE' : [i_seg,:,i_FE,:,:]
id_att_compare = 'Only_Seg'; % 'all', 'Only_Seg', 'Seg_FE'
%-------------------------------------------------------------------------%

% get toolbox
addpath(genpath(fullfile(fileparts(fileparts(fileparts(cd))),'_toolbox')));

% add functions
addpath(genpath(fullfile(cd,'functions')));

% path for processed data
path_research=fileparts(pwd); % parent path which has DB files

% get path
path_DB_process = fullfile(path_research,'DB','DB_processed');
path_DB_raw = fullfile(path_research,'DB','DB_raw');
% load feature set, from this experiment 
tmp = load(fullfile(path_DB_process,name_feat_file)); 
tmp_name = fieldnames(tmp);
feat = getfield(tmp,tmp_name{1}); %#ok<GFLD>

% load feature setfrom another experiment(train-less code)
tmp = load(fullfile(path_DB_process,'feat_set_from_trainless',name_DB_file));
tmp_name = fieldnames(tmp);
feat_DB = getfield(tmp,tmp_name{1}); %#ok<GFLD>

%-----------------------experiment information----------------------------%
[n_seg, n_feat, n_FE, n_trl, n_sub , n_emg_pair] = size(feat); % DB to be analyzed
names_exp = ["화남";"비웃음";"역겨움";"두려움";"행복";"무표정";"슬픔";"놀람"];
n_FE2classfy = length(idx_FE2classfy);
names_exp2classfy = names_exp(idx_FE2classfy);
n_sub_DB = size(feat_DB,5); % Database
idx_sub = 1 : n_sub;
idx_trl = 1 : n_trl;

%
[name_subject,~] = read_names_of_file_in_folder(path_DB_raw);

% feature indexing when using DB of ch4 ver
idx_feat.RMS = 1:4;
idx_feat.WL = 5:8;
idx_feat.SampEN = 9:12;
idx_feat.CC = 13:28;
n_feat = 28;
n_sub_compared = n_sub - 1;
% feat names and indices
name_feat = fieldnames(idx_feat);
idx_feat = struct2cell(idx_feat);
n_ftype = length(name_feat);

%-------------------------------------------------------------------------%

% memory allocation for reults
r_total = cell(n_emg_pair,1);

% get accrucies and output/target (for confusion matrix) with respect to
% subject, trial, number of segment, FE,
for i_emg_pair = 1 : n_emg_pair
    
% memory allocatoin for accurucies
r.acc = zeros(n_seg,n_trl,n_sub,n_transforemd+1);

% memory allocatoin for output and target
r.output_n_target = cell(n_seg,n_trl,n_sub,n_transforemd+1);    

for i_sub = 1 : n_sub
    for i_trl = 1 : n_trl
        
        %display of subject and trial in progress
        fprintf('i_sub:%d i_trial:%d\n',i_sub,i_trl);

        if n_transforemd>=1
        % memory allocation similarily transformed feature set
        feat_t = cell(n_seg,n_FE);
        
        for i_seg = 1 : n_seg
            for i_FE = 1 : n_FE

                % memory allocation feature set from other experiment
                feat_t{i_seg,i_FE} = cell(1,n_ftype);
                
                % you should get access to DB of other experiment with each
                % features
                for i_FeatName = 1 : n_ftype
                    
                    % number of feature of each type
                    n_feat_each = length(idx_feat{i_FeatName});
                    
                    % feat from this experiment
                    feat_ref = feat(i_seg,idx_feat{i_FeatName} ,i_FE,...
                        i_trl,i_sub,i_emg_pair)';
                    
                switch id_DBtype
                case 'DB_own'
                    %---------feat to be compared from this experiment----%
                    % [n_seg:30, n_feat:28, n_FE:8, n_trl:20, n_sub:30, n_emg_pair:3]

                    % compare ohter subject except its own subject
                    idx_sub_compared = countmember(idx_sub,i_sub)==0;
                    switch id_att_compare
                    case 'all'
                        feat_compare = feat(:,idx_feat{i_FeatName},...
                            :,:,idx_sub_compared,i_emg_pair);

                    case 'Only_Seg'
                        feat_compare = feat(i_seg,idx_feat{i_FeatName},...
                            :,:,idx_sub_compared,i_emg_pair);

                    case 'Seg_FE'
                        feat_compare = feat(i_seg,idx_feat{i_FeatName},...
                        i_FE,:,idx_sub_compared,i_emg_pair);
                    end
                    
                    % permutation giving [n_feat, n_FE, n_trl, n_sub ,n_seg]
                    feat_compare = permute(feat_compare,[2 3 4 5 1]);
                    
                    %  size(2):FE, size(5):seg
                    feat_compare = reshape(feat_compare,...
                        [n_feat_each, size(feat_compare,2)*n_trl*n_sub_compared*...
                        size(feat_compare,5)]);
                    
                    % get similar features by determined number of
                    % transformed DB
                    feat_t{i_seg,i_FE}{i_FeatName} = ...
                        dtw_search_n_transf(feat_ref, feat_compare, n_transforemd)';
                    %-----------------------------------------------------%
                    
                    
                   
                case 'DB_other'
                     %---------feat to be compared from other experiment---%
                    % [n_seg:30, n_feat:28, n_FE:8, n_trl:20, n_sub:30]
                    switch id_att_compare
                    case 'all'
                        feat_compare_DB = feat_DB(:,idx_feat{i_FeatName},:,:,:);

                    case 'Only_Seg'
                        feat_compare_DB = feat_DB(i_seg,idx_feat{i_FeatName},:,:,:);

                    case 'Seg_FE'
                        feat_compare_DB = feat_DB(i_seg,idx_feat{i_FeatName},i_FE,:,:);
                    end
                    
                    % permutation giving [n_feat, n_FE, n_trl, n_sub ,n_seg]
                    feat_compare_DB = permute(feat_compare_DB,[2 3 4 5 1]);
                    
                    %  size(2):FE, size(5):seg
                    feat_compare_DB = reshape(feat_compare_DB,...
                        [n_feat_each, size(feat_compare_DB,2)*n_trl*n_sub_DB*...
                        size(feat_compare_DB,5)]);
                    
                    % get similar features by determined number of
                    % transformed DB
                    feat_t{i_seg,i_FE}{i_FeatName} = ...
                        dtw_search_n_transf(feat_ref, feat_compare_DB, n_transforemd)';
                    %-----------------------------------------------------%
                    
                case 'DB_both'
                    %---------feat to be compared from both experiment---%
                    % compare ohter subject except its own subject in this
                    % experiment
                    idx_sub_compared = countmember(idx_sub,i_sub)==0;
                    % [n_seg:30, n_feat:28, n_FE:8, n_trl:20, n_sub:30]
                    switch id_att_compare
                    case 'all'
                        feat_compare = feat(:,idx_feat{i_FeatName},...
                            :,:,idx_sub_compared,i_emg_pair);
                        feat_compare_DB = feat_DB(:,idx_feat{i_FeatName},:,:,:);
                    case 'Only_Seg'
                        feat_compare = feat(i_seg,idx_feat{i_FeatName},...
                            :,:,idx_sub_compared,i_emg_pair);
                        feat_compare_DB = feat_DB(i_seg,idx_feat{i_FeatName},:,:,:);

                    case 'Seg_FE'
                        feat_compare = feat(i_seg,idx_feat{i_FeatName},...
                        i_FE,:,idx_sub_compared,i_emg_pair);
                        feat_compare_DB = feat_DB(i_seg,idx_feat{i_FeatName},i_FE,:,:);
                    end
                    % concatinating both DB
                    feat_both_DB = cat(5,feat_compare,feat_compare_DB);
                    % permutation giving [n_feat, n_FE, n_trl, n_sub ,n_seg]
                    feat_both_DB = permute(feat_both_DB,[2 3 4 5 1]);
                    
                    %  size(2):FE, size(4):sub, size(5):seg
                    feat_both_DB = reshape(feat_both_DB,...
                        [n_feat_each, size(feat_both_DB,2)*n_trl*...
                        size(feat_both_DB,4)*...
                        size(feat_both_DB,5)]);
                    
                    % get similar features by determined number of
                    % transformed DB
                    feat_t{i_seg,i_FE}{i_FeatName} = ...
                        dtw_search_n_transf(feat_ref, feat_both_DB, n_transforemd)';
                    %-----------------------------------------------------%
                end
                end
            end
        end
        
        % arrange feat transformed and target
        % concatinating features with types
        feat_t = cellfun(@(x) cat(2,x{:}),feat_t,'UniformOutput',false);
        end
        % validate with number of transformed DB
        for n_t = 0: n_transforemd
            if n_t >= 1
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
            else
            feat_trans = [];    
            target_feat_trans = [];
            end
            
            % feat for anlaysis
            feat_ref = reshape(permute(feat(:,:,:,i_trl,i_sub,i_emg_pair),...
                [1 3 2]),[n_seg*n_FE,n_feat]);
            target_feat_ref = repmat(1:n_FE,n_seg,1);
            target_feat_ref = target_feat_ref(:);
            
            % get input and targets for train DB
            input_train = cat(1,feat_ref,feat_trans);
            target_train = cat(1,target_feat_ref,target_feat_trans);

            % get input and targets for test DB
            input_test = reshape(permute(feat(:,: ,:,idx_trl~=i_trl,...
                i_sub,i_emg_pair),[1 4 3 2]),[n_seg*(n_trl-1)*n_FE,n_feat]);
            target_test = repmat(1:n_FE,n_seg*(n_trl-1),1);
            target_test = target_test(:);
            
            % get features of determined emotions that you want to classify
            idx_train_samples_2_classify = countmember(target_train,idx_FE2classfy)==1;
            input_train = input_train(idx_train_samples_2_classify,:);
            target_train = target_train(idx_train_samples_2_classify,:);
            
            idx_test_samples_2_classify = countmember(target_test,idx_FE2classfy)==1;
            input_test = input_test(idx_test_samples_2_classify,:);
            target_test = target_test(idx_test_samples_2_classify,:);
            
            % train
            model.lda = fitcdiscr(input_train,target_train);
            
            % test
            output_test = predict(model.lda,input_test);
            
            % reshape ouput_test as <seg, trl, FE>
            output_test = reshape(output_test,[n_seg,(n_trl-1),n_FE2classfy]);
            output_mv_test = majority_vote(output_test,idx_FE2classfy);
            
            % reshape target test for acc caculation
            target_test = repmat(idx_FE2classfy,(n_trl-1),1);
            target_test = target_test(:);
            for i_seg = 1 : n_seg
                ouput_seg = output_mv_test(i_seg,:)';
                r.acc(i_seg,i_trl,i_sub,n_t+1) = ...
                    sum(target_test==ouput_seg)/(n_FE2classfy*(n_trl-1))*100;
                r.output_n_target{i_seg,i_trl,i_sub,n_t+1} = ...
                    [ouput_seg,target_test];
            end
        end
    end
end
% get result
r_total{i_emg_pair} = r;
end

tmp =struct2cell(cell2mat(r_total));
tmp = tmp(1,:);
acc_total = cellfun(@(x) permute(mean(mean(x,2),3),[1 4 2 3]),tmp,'UniformOutput',false);
acc_total = cat(2,acc_total{:});

% set folder for saving
name_folder_saving = ['Result_',name_feat_file,'_',name_DB_file,'_',...
    id_att_compare,'_',id_DBtype,'_n_trans_',num2str(n_transforemd),'_',...
    cat(2,names_exp2classfy{:})];

% set saving folder for windows
path_saving = make_path_n_retrun_the_path(path_DB_process,name_folder_saving);

%---------------------------------save emg_seg----------------------------%
% data structure of accuracy: [i_seg,i_trl,i_sub,n_t+1]
save(fullfile(path_saving,'result'),'acc_total','r_total');
% plot with emg_pair
for i_emg_pair = 1 : n_emg_pair
    tmp = r_total{i_emg_pair};
    tmp = permute(mean(mean(tmp.acc,2),3),[1 4 2 3]);
    figure;
    plot(tmp)
end


% plot confusion matrix with specific subejct of a emg pair
for i_emg_pair = 1 : n_emg_pair
    tmp = r_total{i_emg_pair};
    tmp = tmp.output_n_target(15,:,2,:);
    tmp = cat(1,tmp{:});
    
    output_tmp = full(ind2vec(tmp(:,1)'));
    target_tmp = full(ind2vec(tmp(:,2)'));
    
    tmp = countmember(1:max(idx_FE2classfy),idx_FE2classfy)==0;
    output_tmp(tmp,:) = [];
    target_tmp(tmp,:) = [];
    
    [~,mat_conf,idx_of_samps_with_ith_target,~] = ...
        confusion(target_tmp,output_tmp);
    figure(i_emg_pair);
    plotConfMat(mat_conf, names_exp2classfy)
%     mat_n_samps = cellfun(@(x) size(x,2),idx_of_samps_with_ith_target);
%     mat_n_samps(logical(eye(size(mat_n_samps)))) = 0;
%     fn_sum_of_each_class = sum(mat_n_samps,1);
end


for i_emg_pair = 1 : n_emg_pair
    tmp = r_total{i_emg_pair};
    tmp = permute(mean(tmp.acc(15,:,:,:),2),[3 4 1 2]);
    figure;
    bar(tmp)
    mean(tmp)
end




%-------------------------------------------------------------------------%


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


