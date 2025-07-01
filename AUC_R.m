function [S,R] = AUC_R(X,Xi,varargin)
% Compute the log scaled area under the curve R_NX(K).
% Input: 
% X: n*D high-dim data;
% Xi: n*d low-dim representations;
% opt: 1 for the Mahanobis distance, else for the Euclidean distance, 1 as default; 
% n_prop: proportion of individuals used in computation,
%       a value in (0,1], lower value for faster but less accurate computation, 1 as default;
% Output:
% S: AUC of log R_NX;
% R: 1*n-2 vector of R_NX(K).

% Author: Ruoxu Tan; date: 2025/Jul; Matlab version: R2023b.
p = inputParser;
addParameter(p,'opt',1);
addParameter(p,'n_prop',1); 
parse(p,varargin{:});
opt = p.Results.opt;
n_prop = p.Results.n_prop;

if size(X,1) ~= size(Xi,1)
    error('The dimensions of the input do not macth.')
end
n = size(X,1);

Rank_X = zeros(n);
Rank_Xi = Rank_X; 

if opt == 1
    if rank(X) >= size(X,2)
        Cov_X = (X-mean(X,1))' * (X-mean(X,1)) ./ n;
        X = X * Cov_X^(-1/2);
    else
        Var_X = diag((X-mean(X,1))' * (X-mean(X,1)) ./ n);
        X(:,Var_X==0) = []; 
        Var_X = Var_X(Var_X ~= 0);
        sd_inv = Var_X.^(-1/2);
        X = X .* sd_inv';
    end
    
    Cov_Xi = (Xi-mean(Xi,1))' * (Xi-mean(Xi,1)) ./ n;
    Xi = Xi * Cov_Xi^(-1/2); 
end

for i = 1:n
    [~,ind_X_i] = sort(sum((X-X(i,:)).^2,2));
    [~,ind_Xi_i] = sort(sum((Xi-Xi(i,:)).^2,2)); 
    
    [~,Rank_X(i,:)] = sort(ind_X_i);
    [~,Rank_Xi(i,:)] = sort(ind_Xi_i);
end

Rank_X = Rank_X - 1;
Rank_Xi = Rank_Xi - 1;

m = round(n*n_prop);
if m < 3
    m = 3;
end
n_sc = randperm(n,m);
R = zeros(1,m-2);
for k = 1:m-2
    Q_k = sum((Rank_X(n_sc,:) >= 1 & Rank_X(n_sc,:) <= k) & (Rank_Xi(n_sc,:) >= 1 & Rank_Xi(n_sc,:) <= k),'all') / (k*m);
    R(k) = ((n-1)*Q_k-k) / (n-1-k);
end

scale = 1:m-2;   
S = trapz(log(scale),R);

end
    