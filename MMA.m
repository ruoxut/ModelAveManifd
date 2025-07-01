function [Xi_ave,W] = MMA(X,Xi_all,varargin)
% Model averaging for manifold learning outcomes.
% Input:
% X: n*D high-dim data;
% Xi_all: 1*m cell, each cell contains a n*d matrix of manifold learning outcomes;
% n_prop: proportion of individuals used in computation,
%       a value in (0,1], lower value for faster computation, 1 as default;
% Output:
% Xi_ave: n*d averaged manifold learning outcomes;
% W: 1*m weighting vector.

% Author: Ruoxu Tan; date: 2025/Jul; Matlab version: R2023b.
p = inputParser; 
addParameter(p,'n_prop',1); 
parse(p,varargin{:}); 
n_prop = p.Results.n_prop;

if size(Xi_all,1) ~= 1
    error('Input dimensions do not match.')
end

m = size(Xi_all,2);
n = size(X,1);

% Standardizing
for i = 1:m
    Xi_all{1,i} = Xi_all{1,i}-mean(Xi_all{1,i},1);
    Cov_i = Xi_all{1,i}' * Xi_all{1,i} ./ n;
    Xi_all{1,i} = Xi_all{1,i} * Cov_i^(-1/2);
end

% Optimization
Rank_X = zeros(n);

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

for i = 1:n
    [~,ind_X_i] = sort(sum((X-X(i,:)).^2,2));
    [~,Rank_X(i,:)] = sort(ind_X_i);
end

Rank_X = Rank_X - 1;

fun = @(W) AUC_ave(W,Rank_X,Xi_all,n_prop); 

lb = zeros(1,m);
ub = ones(1,m);
options = optimoptions(@fmincon,'Algorithm','sqp','SpecifyConstraintGradient',true);
W_ini = ones(1,m)./m;  

problem = createOptimProblem('fmincon','objective',fun,'lb',lb,'ub',ub,...
                             'x0',W_ini,'nonlcon',@mycon,'options',options);
ms = MultiStart('UseParallel',true,'StartPointsToRun','bounds-ineqs');
W = run(ms,problem,32);
                     
Xi_ave = Xi_all{1,1}.*W(1,1);
for i = 2:m
    Xi_ave = Xi_ave + Xi_all{1,i}.*W(1,i);
end

end

function [AUC_res_neg] = AUC_ave(W,Rank_X,Xi_all,n_prop) 
m = size(Xi_all,2);
Xi_ave = Xi_all{1,1}.*W(1,1);
n = size(Xi_ave,1);

for i = 2:m
    Xi_ave = Xi_ave + Xi_all{1,i}.*W(1,i);
end

Rank_Xi = zeros(n); 
Cov_Xi = (Xi_ave-mean(Xi_ave,1))' * (Xi_ave-mean(Xi_ave,1)) ./ n;
Xi_ave = Xi_ave * Cov_Xi^(-1/2); 

for i = 1:n
    [~,ind_Xi_i] = sort(sum((Xi_ave-Xi_ave(i,:)).^2,2)); 
    [~,Rank_Xi(i,:)] = sort(ind_Xi_i);
end

Rank_Xi = Rank_Xi - 1;
m_n = round(n*n_prop);
if m_n < 3
    m_n = 3;
end
n_sc = randperm(n,m_n);
R = zeros(1,m_n-2);
for k = 1:m_n-2
    Q_k = sum((Rank_X(n_sc,:) >= 1 & Rank_X(n_sc,:) <= k) & (Rank_Xi(n_sc,:) >= 1 & Rank_Xi(n_sc,:) <= k),'all') / (k*m_n);
    R(k) = ((n-1)*Q_k-k) / (n-1-k);
end

scale = 1:m_n-2;   
S = trapz(log(scale),R);
AUC_res_neg = -S; 

end

function [c,ceq,c_g,ceq_g] = mycon(W)
c = [];  
ceq = sum(W)-1;
c_g = [];
ceq_g = ones(length(W),1);
end
