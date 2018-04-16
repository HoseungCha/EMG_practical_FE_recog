%----------------------------------------------------------------------
% Calculate Window size & Window increase 
%---------------------------------------------------------------------
%
% �Էº���  
% original_sam_rate: ���� sampling rate 
% convert_sam_rate: �ٲٷ��� �ϴ� sampling rate
% overlap: �� % overlap �Ϸ�����
% ex) 2048Hz�� 10Hz�� 50% overlap�Ϸ��� 2048,10,50 �������

function [winsize,wininc] = calculate_window (original_sam_rate,convert_sam_rate, overlap)

    wininc = floor(original_sam_rate / ((convert_sam_rate-1) * (100/overlap-1) +1));
    winsize = floor(100 / overlap * wininc);

end