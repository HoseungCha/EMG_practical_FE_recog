%----------------------------------------------------------------------
% EMG speech onset 부분 추출하여 합치는 코드
%----------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%---------------------------------------------------------------------
clear; close all; clc
% add toolbox
addpath(genpath(fullfile(fileparts(fileparts(fileparts(pwd))),'_matlab_toolbox')));
% add functions I developed
addpath(genpath(fullfile(cd,'functions')));
% get parent path of this ode
path_parent=fileparts(pwd);

% 실험 정보
names_exp = ["화남";"비웃음";"역겨움";"두려움";"행복";"무표정";"슬픔";"놀람"];
n_pair = 1;
% path_main= 'E:\OneDrive_Hanyang\연구\EMG_FE_recognition_emotion\코드'; % main path
path_saved = fullfile(path_parent,'DB','ProcessedDB');%,...
%     'len_win_0.1000_SP_win_0.1000'); % saving path
load(fullfile(path_saved,'feat_set_combined_of_tless_prac_seg_60_using_ch4')); % load saved features
features=feat_set_combined; clear feat_set_combined;
% 분류 감정 선택
idx_exp2use =[1,5,7,8]; % 전체:"화남";"비웃음";"역겨움";"두려움";"행복";"무표정";"슬픔";"놀람"
% idx_exp2use = [1,2,3,5,6,7,8]; % <두려움제거>
% idx_exp2use = [1,2,3,4,5,6,7]; %  <놀람 제거>
% idx_exp2use = [1,2,3,5,7,8]; % <두려움 무표정 제거>
% 화남, 놀람, 행복(기쁨), 슬픔(역겨움) <교수님 선정>

disp(names_exp(idx_exp2use)) % 확인
features = features(:,:,idx_exp2use,:,:);

% DB 정보
[n_seg, n_feat,n_exp,n_trl,n_sub] = size(features);

% feature indexing
idx_feat_CC = 1:16;
idx_feat_RMS = 17:20;
idx_feat_SampEN = 21:24;
idx_feat_WL = 25:28;

% get indices of trials
Idx_trial = 1 : n_trl;

% feature 별로 추출
F.CC = features(:,idx_feat_CC,:,:,:);
F.RMS = features(:,idx_feat_RMS,:,:,:);
F.SampEN = features(:,idx_feat_SampEN,:,:,:);
F.WL = features(:,idx_feat_WL,:,:,:);
clear idx_feat_CC idx_feat_RMS idx_feat_SampEN idx_feat_WL

% sturct to cell and naming each feature
F_name = fieldnames(F);
F_cell = struct2cell(F);

% % window segments, number of trials(repetition), number of subjects
% [N_seg, ~, ~, N_trl, N_sub] = size(F.WL);


% memory allocations for accurucies of classification algorithms
acc.svm = zeros(n_seg,n_trl,n_sub);
acc.lda = zeros(n_seg,n_trl,n_sub);
acc.knn = zeros(n_seg,n_trl,n_sub);

N_comp = 1; N_Total_comp=1;
for N_expatpair = 4  % choose numbe of feature pairs. % 참고: feature 4개 사용했을 때 결과 좋음
    idx_F = nchoosek(1:length(F_name),N_expatpair);
    
    % search similar features (transformed features) from Dataset by using DTW
    for i_feat = 1 : size(idx_F,1)
        % get each feat size
        feat_size = cellfun(@(x) size(x,2),F_cell);
        % concatinating dataset by features
        temp_feat = cat(2,F_cell{:});
        
        % reject an expression which evoke confusion
%         idx2reject = 4; % 참고: 4: Fear, 3: Disgusted, 2개 표정이 confusion 많이 일으킴
%         temp_feat(:,:,idx2reject,:,:) = [];
%         [~, N_expat, N_exp, ~, ~] = size(temp_feat);
        
        % training 수를 줄여가면서 Validation
        % train DB를 20가지중에 선택할 때, 랜덤하게 선택 총 반복횟수 정하기
        load('pairset_new.mat'); 
        
        for n_pair = 1 : n_pair;
        pair = pairset_new{n_pair};
        
        % memory allocation for results
        pred.svm = cell(n_sub,n_trl,N_Total_comp+1);
        pred.lda = cell(n_sub,n_trl,N_Total_comp+1);
        pred.knn = cell(n_sub,n_trl,N_Total_comp+1);
        pred.label = cell(n_sub,n_trl,N_Total_comp+1);
        fin_pred.svm = cell(n_sub,n_trl,N_Total_comp+1);
        fin_pred.lda = cell(n_sub,n_trl,N_Total_comp+1);
        fin_pred.knn = cell(n_sub,n_trl,N_Total_comp+1);
        pred_n_label = cell(n_sub,n_trl,N_Total_comp+1);
        
        for N_comp = 0:N_Total_comp
%         pred_n_label__=cell(N_sub,1);
%         c_sub = 0;
        for i_sub = 31 : n_sub
%         for i_sub = 31
%         c_sub = c_sub
%         fprintf('N_comp:%d i_sub: %d \n',N_comp,i_sub);
%         pred_n_label_ = cell(N_trl,1);
        for i_trl = 1:n_trl
%         for i_trl = 1
            % get train DB
            train = temp_feat(:,:,:,Idx_trial==pair(i_trl,n_pair),i_sub);
            % get permutation of dimesion of train DB
            train = permute(train,[1 3 2]);
            train = reshape(train, [n_seg*n_exp, size(train,3)]);
            label = repmat(1:n_exp,[n_seg,1]); label = label(:);
            % prepare user DB and similar DB from database
            if (N_comp>0) %%%%%%%%%get similar DB%%%%%%%%%%%%%%%%%%%%%%%%%%
                d_similar = cell(size(idx_F,2),1); labe_d = cell(size(idx_F,2),1);
                for i_featp = 1 : size(idx_F,2)
                    count = 1;
                    d_similar{i_featp} = zeros(n_exp*n_seg,feat_size(i_featp));
                    labe_d{i_featp} = zeros(length(d_similar{i_featp}),1);
                    feat_name  = F_name{idx_F(i_feat,i_featp)};
                    load(fullfile('E:\Hanyang\연구\EMG_TrainLess_Expression\코드\DB\dist\T5_feat_set_combined_of_tless_prac_seg_60_using_ch4',...
                        ['T_',num2str(i_sub),'_',...
                        num2str(i_trl),'_',feat_name,'_5.mat']));
                    T = T(:,idx_exp2use);
                    for i_FE = 1 : n_exp
                        for i_seg = 1 : n_seg
                            for i_comp = 1 : N_comp
                                d_similar{i_featp}(count,:) = T{i_seg,i_FE}(:,i_comp)';
                                labe_d{i_featp}(count) = i_FE;
                                count = count + 1 ;
                            end
                        end
                    end
                    d_similar{i_featp} = d_similar{i_featp}';
                end

                t = [train;cell2mat(d_similar)']; l = [label;labe_d{1}];
            else%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                t = train;  l = label;
            end
            % train machine lerning model
%             model.svm= svmtrain(l , t,'-s 1 -t 0 -b 1 -q');
            model.lda = fitcdiscr(t,l);
%             model.knn = fitcknn(t,l,'NumNeighbors',5);
            % test data
            test =  temp_feat(:,:,:,~logical(countmember(Idx_trial,pair(i_trl,:))),i_sub);
            test = permute(test,[1 4 3 2]);
            test = reshape(test, [n_seg*(n_trl-n_pair)*n_exp, size(test,4)]);
            label = repmat(1:n_exp,[n_seg*(n_trl-n_pair),1]); label = label(:);
            % test

%             pred.svm{i_sub,i_trl,N_comp+1} = svmpredict(label,test, model.svm,'-b 1 -q');
            pred.lda{i_sub,i_trl,N_comp+1} = predict(model.lda,test);
%             pred.knn{i_sub,i_trl,N_comp+1} = predict(model.knn,test);
            pred.label{i_sub,i_trl,N_comp+1} = label;

%             pred_svm = reshape(pred.svm{i_sub,i_trl,N_comp+1},[N_seg,((N_trl-n_pair)),N_exp]);
            pred_lda = reshape(pred.lda{i_sub,i_trl,N_comp+1},[n_seg,((n_trl-n_pair)),n_exp]);
%             pred_knn = reshape(pred.knn{i_sub,i_trl,N_comp+1},[N_seg,((N_trl-n_pair)),N_exp]);

%             fin_pred.svm{i_sub,i_trl,N_comp+1} = majority_vote(pred_svm);
            fin_pred.lda{i_sub,i_trl,N_comp+1} = majority_vote(pred_lda);
%             fin_pred.knn{i_sub,i_trl,N_comp+1} = majority_vote(pred_knn);
             
%           % reshape target test for acc caculation
            target_test = repmat(1:n_exp,(n_trl-1),1);
            target_test = target_test(:);
            for n_seg = 1 : n_seg
                ouput_seg = fin_pred.lda{i_sub,i_trl,N_comp+1} (n_seg,:)';
                acc.lda(n_seg,i_trl,i_sub,N_comp+1) = ...
                    sum(target_test==ouput_seg)/(n_exp*(n_trl-1))*100;
                pred_n_label{n_seg,i_trl,i_sub,N_comp+1} = ...
                    [ouput_seg,target_test];
                
%                 acc.svm(n_seg,i_trl,i_sub,N_comp+1) = sum(repmat((1:N_exp)',...
%                     [(N_trl-n_pair),1]) == fin_pred.svm{i_sub,i_trl,N_comp+1}(:,n_seg))...
%                     /((N_trl-n_pair)*N_exp)*100;
%                 acc.lda(n_seg,i_trl,i_sub,N_comp+1) = sum(repmat((1:N_exp)',...
%                     [(N_trl-n_pair),1]) == fin_pred.lda{i_sub,i_trl,N_comp+1}(n_seg,:)')...
%                     /((N_trl-n_pair)*N_exp)*100;
%                 pred_n_label{n_seg,i_trl,i_sub,N_comp+1} = [fin_pred.lda{i_sub,i_trl,N_comp+1}(:,n_seg),...
%                     repmat((1:N_exp)',[(N_trl-n_pair),1])];
%                 acc.knn(n_seg,i_trl,i_sub,N_comp+1) = sum(repmat((1:N_exp)',...
%                     [(N_trl-n_pair),1]) == fin_pred.knn{i_sub,i_trl,N_comp+1}(:,n_seg))...
%                     /((N_trl-n_pair)*N_exp)*100;
            end
        end
%         figure(n_pair);plot(permute(mean(mean(acc.lda(:,:,31:i_sub,1:(N_comp+1)),3),2),[1 4 2 3]));drawnow;
        end
        end
        
        % plot and analyze of confusion matrix
%         n_seg = 30;
%         for i_sub = 1 : N_sub
%             pred_n_label_vec = cat(1,pred_n_label{n_seg,:,i_sub});
%             outputs = full(ind2vec(pred_n_label_vec(:,1)',N_exp));
%             targets = full(ind2vec(pred_n_label_vec(:,2)',N_exp));
%             [c,cm,ind,per] = confusion(targets,outputs);
%             figure(i_sub);
%             plotConfMat(cm,  names_exp(idx_exp2use))
%         end
%         for n_seg = 1:10:N_seg
%         pred_n_label_vec = cat(1,pred_n_label{n_seg,:,:});
%         outputs = full(ind2vec(pred_n_label_vec(:,1)',N_exp));
%         targets = full(ind2vec(pred_n_label_vec(:,2)',N_exp));
% %             plotconfusion(targets,outputs,names_word)
%         [c,cm,ind,per] = confusion(targets,outputs);
%         figure;
%         plotConfMat(cm, names_exp(idx_exp2use))
%         sp = cellfun(@(x) size(x,2),ind)
%         sp(logical(eye(size(sp)))) = 0;
%         FN_sum_of_each_class = sum(sp,1);
%         end
        end
    end
end
%% 결과 정리
size(acc.lda)
acc.lda = acc.lda(:,:,31:end,:);
% 피험자 및 사용한 유사 DB갯수로 정의
result = permute(mean(acc.lda(30,:,:,:),2),[3 4 1 2]);
bar(result)
mean(result)




