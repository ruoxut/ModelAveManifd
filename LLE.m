function [ Xi ] = LLE(X,K,d)
% Local linear embedding.
% Input: 
% X: n*D data matrix;
% K: number of nearest neighbours;
% d: intrinsic dimension;
% Output:
% Xi: n*d low-dim representations.

% Author: Ruoxu Tan; date: 2025/Jul; Matlab version: R2023b.

n = size(X,1);
W = zeros(n); 

for i = 1:n
    dis_i = vecnorm(X'-X(i,:)' );
    [~,ind_i] = sort(dis_i); 
    
    z = X(ind_i(2:K+1),:) - X(i,:);
    C_i = z * z' ;
    if cond(C_i) > 1e3
        C_i = C_i + 0.01.*norm(C_i)./K.*eye(K);
    end
    W_i = C_i\ones(K,1);
    W_i = W_i/sum(W_i);
    W(i,ind_i(2:K+1)) = W_i;
end

M = eye(n) - W - W' + W'*W;

[V,D] = eig(M);
[~,ind] = sort(diag(D));
V = V(:,ind);
Xi = V(:,2:d+1);
