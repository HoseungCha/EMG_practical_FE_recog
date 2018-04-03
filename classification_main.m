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
parentdir=(fileparts(pwd));
addpath(genpath(fullfile(parentdir,'functions')));
% 실험 정보
% Trg_Inform = {"화남 ",1,1;"비웃음",1,2;"역겨움",1,3;"두려움",1,4;"행복",1,5;"무표정",1,6;"슬픔",1,7;"놀람",2,1;"키스",2,2};
names_exp = ["화남";"비웃음";"역겨움";"두려움";"행복";"무표정";"슬픔";"놀람";"키스"];

% path_main= 'E:\OneDrive_Hanyang\연구\EMG_FE_recognition_emotion\코드'; % main path
path_saved = fullfile(parentdir,'DB','ProcessedDB');%,...
%     'len_win_0.1000_SP_win_0.1000'); % saving path
load(fullfile(path_saved,'feat_set')); % load saved features
% 분류 감정 선택
idx_exp2use = [1,2,3,5,7,8]; %["화남";"비웃음";"역겨움";"두려움";"행복";"무표정";"슬픔";"놀람";
idx_exp2use = 1 : 8
names_exp(idx_exp2use) % 확인
Features = Features(:,:,idx_exp2use,:,:);

% DB 정보
[N_seg, N_feat,N_exp,N_trl,N_sub] = size(Features);

% feature indexing
idx_feat_CC = 1:16;
idx_feat_RMS = 17:20;
idx_feat_SampEN = 21:24;
idx_feat_WL = 25:28;

% feature 별로 추출
F.CC = Features(:,idx_feat_CC,:,:,:);
F.RMS = Features(:,idx_feat_RMS,:,:,:);
F.SampEN = Features(:,idx_feat_SampEN,:,:,:);
F.WL = Features(:,idx_feat_WL,:,:,:);
clear idx_feat_CC idx_feat_RMS idx_feat_SampEN idx_feat_WL

% sturct to cell and naming each feature
F_name = fieldnames(F);
F_cell = struct2cell(F);

% % window segments, number of trials(repetition), number of subjects
% [N_seg, ~, ~, N_trl, N_sub] = size(F.WL);


% memory allocations for accurucies of classification algorithms
acc.svm = zeros(N_seg,N_trl,N_sub);
acc.lda = zeros(N_seg,N_trl,N_sub);
acc.knn = zeros(N_seg,N_trl,N_sub);

N_comp = 1; N_Total_comp=0;
for N_expatpair = 1  % choose numbe of feature pairs. % 참고: feature 4개 사용했을 때 결과 좋음
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
        
        for n_pair = 1;
        pair = pairset_new{n_pair};
        % get indices of trials
        Idx_trial = 1 : N_trl;
        
        % IDX_comp = 10 : 10 : 100; IDX_comp = [1, IDX_comp];
        % IDX_comp = 1 : 100;
        % N_Total_comp = length(IDX_comp);
        
        % memory allocation for results
        pred.svm = cell(N_sub,N_trl,N_Total_comp+1);
        pred.lda = cell(N_sub,N_trl,N_Total_comp+1);
        pred.knn = cell(N_sub,N_trl,N_Total_comp+1);
        pred.label = cell(N_sub,N_trl,N_Total_comp+1);
        fin_pred.svm = cell(N_sub,N_trl,N_Total_comp+1);
        fin_pred.lda = cell(N_sub,N_trl,N_Total_comp+1);
        fin_pred.knn = cell(N_sub,N_trl,N_Total_comp+1);
        pred_n_label = cell(N_sub,N_trl,N_Total_comp+1);
        
%         for N_comp = 0 : N_Total_comp
%         for N_comp = N_Total_comp
%         pred_n_label__=cell(N_sub,1);
        for i_sub = 1 : N_sub
%         fprintf('N_comp:%d i_sub: %d \n',N_comp,i_sub);
%         pred_n_label_ = cell(N_trl,1);
        for i_trl = 1:N_trl
            % get train DB
            train = temp_feat(:,:,:,Idx_trial==pair(i_trl,n_pair),i_sub);
            % get permutation of dimesion of train DB
            train = permute(train,[1 3 2]);
            train = reshape(train, [N_seg*N_exp, size(train,3)]);
            label = repmat(1:N_exp,[N_seg,1]); label = label(:);
            % prepare user DB and similar DB from database
            if (N_comp>0) %%%%%%%%%get similar DB%%%%%%%%%%%%%%%%%%%%%%%%%%
                d_similar = cell(size(idx_F,2),1); labe_d = cell(size(idx_F,2),1);
                for i_featp = 1 : size(idx_F,2)
                    count = 1;
                    d_similar{i_featp} = zeros(N_exp*N_seg,feat_size(i_featp));
                    labe_d{i_featp} = zeros(length(d_similar{i_featp}),1);
                    feat_name  = F_name{idx_F(i_feat,i_featp)};
                    load(fullfile('E:\OneDrive_Hanyang\연구\EMG_TrainLess_Expression\코드\DB\dist',...
                        'T_100',['T_',num2str(i_sub),'_',...
                        num2str(i_trl),'_',feat_name,'_100.mat']));
%                     T(:,idx2reject) = [];
                    for i_FE = 1 : N_exp
                        for i_seg = 1 : N_seg
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
            test = reshape(test, [N_seg*(N_trl-n_pair)*N_exp, size(test,4)]);
            label = repmat(1:N_exp,[N_seg*(N_trl-n_pair),1]); label = label(:);
            % test

%             pred.svm{i_sub,i_trl,N_comp+1} = svmpredict(label,test, model.svm,'-b 1 -q');
            pred.lda{i_sub,i_trl,N_comp+1} = predict(model.lda,test);
%             pred.knn{i_sub,i_trl,N_comp+1} = predict(model.knn,test);
            pred.label{i_sub,i_trl,N_comp+1} = label;

%             pred_svm = reshape(pred.svm{i_sub,i_trl,N_comp+1},[N_seg,((N_trl-n_pair)),N_exp]);
            pred_lda = reshape(pred.lda{i_sub,i_trl,N_comp+1},[N_seg,((N_trl-n_pair)),N_exp]);
%             pred_knn = reshape(pred.knn{i_sub,i_trl,N_comp+1},[N_seg,((N_trl-n_pair)),N_exp]);

%             fin_pred.svm{i_sub,i_trl,N_comp+1} = majority_vote(pred_svm);
            fin_pred.lda{i_sub,i_trl,N_comp+1} = majority_vote(pred_lda);
%             fin_pred.knn{i_sub,i_trl,N_comp+1} = majority_vote(pred_knn);
             
%             pred_n_label = cell(N_seg,1);
            for n_seg = 1 : N_seg
%                 acc.svm(n_seg,i_trl,i_sub,N_comp+1) = sum(repmat((1:N_exp)',...
%                     [(N_trl-n_pair),1]) == fin_pred.svm{i_sub,i_trl,N_comp+1}(:,n_seg))...
%                     /((N_trl-n_pair)*N_exp)*100;
                acc.lda(n_seg,i_trl,i_sub,N_comp+1) = sum(repmat((1:N_exp)',...
                    [(N_trl-n_pair),1]) == fin_pred.lda{i_sub,i_trl,N_comp+1}(:,n_seg))...
                    /((N_trl-n_pair)*N_exp)*100;
                pred_n_label{n_seg,i_trl,i_sub,N_comp+1} = [fin_pred.lda{i_sub,i_trl,N_comp+1}(:,n_seg),...
                    repmat((1:N_exp)',[(N_trl-n_pair),1])];
%                 acc.knn(n_seg,i_trl,i_sub,N_comp+1) = sum(repmat((1:N_exp)',...
%                     [(N_trl-n_pair),1]) == fin_pred.knn{i_sub,i_trl,N_comp+1}(:,n_seg))...
%                     /((N_trl-n_pair)*N_exp)*100;
            end
        end
        figure(n_pair);plot(permute(mean(mean(acc.lda(:,:,1:i_sub,1:(N_comp+1)),3),2),[1 4 2 3]));drawnow;
        end
        
        n_seg = 30;
%         for i_sub = 1 : N_sub
%             pred_n_label_vec = cat(1,pred_n_label{n_seg,:,i_sub});
%             outputs = full(ind2vec(pred_n_label_vec(:,1)',N_exp));
%             targets = full(ind2vec(pred_n_label_vec(:,2)',N_exp));
%             [c,cm,ind,per] = confusion(targets,outputs);
%             figure(i_sub);
%             plotConfMat(cm,  names_exp(idx_exp2use))
%         end
        
%         for n_seg = 1:10:N_seg
        pred_n_label_vec = cat(1,pred_n_label{n_seg,:,:});
        outputs = full(ind2vec(pred_n_label_vec(:,1)',N_exp));
        targets = full(ind2vec(pred_n_label_vec(:,2)',N_exp));
%             plotconfusion(targets,outputs,names_word)
        [c,cm,ind,per] = confusion(targets,outputs);
        figure;
        plotConfMat(cm, names_exp(idx_exp2use))
        sp = cellfun(@(x) size(x,2),ind)
        sp(logical(eye(size(sp)))) = 0;
        FN_sum_of_each_class = sum(sp,1);

%         end
        end
%         save(fullfile(cd,'result',[temp_str,'._FearR_mat']),'acc','pred','fin_pred');
    end
end



% % train/test indexing
% idx_rand = randperm(numel(feat_set));
% idx_train = idx_rand(1: 63);
% idx_test =  idx_rand(64: end);
% 
% % get train DB & label
% feat_train = feat_set(idx_train);
% feat_train = cell2mat(cat(1,feat_train{:}));
% label = repmat((1: N_exp)',[63,N_seg])';
% label = label(:);
% 
% %train LDA
% model_LDA = fitcdiscr(feat_train, label);
% 
% %test LDA
% feat_test = feat_set(idx_test);
% for i = 1 : numel(feat_test)
%     temp = feat_test{i};
%     temp = cat(3,temp{:});
% 
%     [temp_pred,~,~] = predict(model_LDA, temp_f);
%     [temp_pred,~,~] = svmpredict(temp_l,temp_f, model);
%                 
%     % final decision using majoriy voting
%     for n_seg = 1 : N_Seg
%         maxv = zeros(N_exp,N_trial-n_pair); final_predict = zeros(N_exp,N_trial-n_pair);
%         for i = 1 : N_exp
%             for j = 1 : N_trial-n_pair
%                 [maxv(i,j),final_predict(i,j)] = max(countmember(1:N_exp,temp_pred(i,1:n_seg,j)));
%             end
%         end
%         final_predict = final_predict(:);
%         acc(i_sub,i_tral,n_seg) = sum(repmat((1:N_exp)',[N_trial-n_pair,1])==final_predict)/(N_exp*N_trial-N_exp*n_pair)*100;
%     end
%     
% end
% 
% 
% 
% cellfun(@(x) size(x,1),feat_set);
% cat(1,feat_set{:});
% 
% c