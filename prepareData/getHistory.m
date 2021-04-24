function R = getHistory(p, Xj)

% for each feature, say Xj = AE in hrs, compute endWindow rolling mean

% p: vector that store past hours
% p = [3,6,12,24,36,48] as 0-3, 0-6, 0-12,... past hours
% Xj: n x 1 feature, such as AE
% R: current data and rolling history of past 0-p(1) hrs, past 0-p(2)hrs

% need mean window end at the current position (not centerd as the movmean)
% col 1        , col 2       , col 3       , col 4
% curr time: 48, roll(45:48) , roll(32:48) , roll(36:48)..
% curr time: p ,  roll(p-3:p), roll (p-6,p), roll(p-12,p),...

numCol = length(p);
numData = length(Xj);
R = zeros(numData, numCol);

for i = 1:numCol
    m = p(i)+1;
    centeredWindow = movmean(Xj,m);
    endWindow = NaN(numData,1);
    if mod(m,2)==0
        % movmean with history of m hours, window is centered about the
        % current and previous elements
        % m even, m/2 integer, shift down m/2 elements
        movedown = m/2;
    else
        % m odd, shift down ceil(m/2) elements
        movedown = ceil(m/2);
    end
    endWindow(movedown:end) = centeredWindow(1:(numData-movedown+1));
    endWindow(1:m-1) = nan;
    R(:,i) = endWindow;
end

% for debug
% R = [Xj,R];
end