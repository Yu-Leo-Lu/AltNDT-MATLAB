function [W,b] = initW1b1(Children, IsBranchNode)
%sample:
% Children =
% 
%      2     3
%      4     5
%      6     7
%      0     0
%      8     9
%      0     0
%     10    11
%      0     0
%      0     0
%      0     0
%      0     0

% IsBranchNode =
% 
%   11×1 logical array
% 
%    1
%    1
%    1
%    0
%    1
%    0
%    1
%    0
%    0
%    0
%    0

node = length(IsBranchNode);
K = floor(length(IsBranchNode)/2);
[i,~] = find(Children==11);
W = zeros(node, i);
while node>=1
    if IsBranchNode(node)==0
        i = 0;
        leaf = node;
        while i~=1
            [i,j] = find(Children==leaf);
            if j==1
                %genWb1 error: last few nodes are not always leaves!
                W(node,i)=1;
            else
                W(node,i)=-1;
            end
            leaf=i;
        end
    end
    
    node = node-1;
end
W = W(any(W,2),:);
W = W(:,any(W,1));

b = zeros(K+1,1);
for i = 1:K+1
    b(i) = -(sum(W(i,:)==1)-1/2);
end
end