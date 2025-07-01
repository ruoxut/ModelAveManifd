function [ Xi,G_s,ind_max ] = Isomap( X,K,d )
% Isomap dimension reduction.
% Input: 
% X: n*D data matrix;
% K: number of nearest neighbours;
% d: intrinsic dimension;
% Output:
% Xi: n*d low-dim representations;
% G_s: n*n proximity graph. 
% ind_max: index set of the largest component.

% Author: Ruoxu Tan; date: 2023/Jan/29; Matlab version: R2020a.

n = size(X,1);

%% Proximity graph
G = zeros(n);
for i = 1:n
    for j = i+1:n
        G(i,j) = norm(X(i,:)-X(j,:));
        G(j,i) = G(i,j);
    end
    
    [~,ind] = sort(G(i,:));
    G(i,ind(K+2:end)) = Inf;
end

for i = 1:n
    for j = i+1:n
        G(i,j) = min([G(i,j),G(j,i)]);
        G(j,i) = G(i,j);
    end
end

% Shortest path graph 
G_s = zeros(n); 
for i = 1:n
    Dis = zeros(1,n);
    Pre = zeros(1,n);
    Dis(:) = Inf;
    Dis(i) = 0;
    Q = 1:n;
    while ~isempty(Q)
        [~,j] = min(Dis(Q));
        u = Q(j);
        Q(j) = [];
        for k = 1:n
            if k ~= u 
                alt = Dis(u)+G(u,k);
                if alt < Dis(k)
                Dis(k) = alt;
                Pre(k) = u;
                end
            end
        end
    end
    G_s(i,:) = Dis; 
end 

% Find the largest connected component  
G_com = zeros(1,n);

if sum(sum(isinf(G_s))) == 0
    G_com = G_com + 1;
else
    n_c = 1;
    for i = 1:n        
        if G_com(1,i) == 0
            G_com = DFS(G_s,G_com,n_c,i);
            n_c = n_c+1;
        end
    end
end

n_com = max(G_com);
if n_com > 1
    warning('The proximity graph is not connected and the dimension reduction is performed on the largest connected component.')
end

size_com = zeros(1,n_com);
for i = 1:n_com
    size_com(i) = sum(find(G_com==i));
end
[~,com_max] = max(size_com);
ind_max = G_com == com_max;
G_s = G_s(ind_max,ind_max);
 
% Perform MDS on the largest component 
D_sq = G_s.^2;
n_i = size(D_sq,1);
G_cen = -0.5 * (eye(n_i)-(1/n_i) * ones(n_i,1) * ones(1,n_i)) * D_sq * (eye(n_i)-(1/n_i) * ones(n_i,1) * ones(1,n_i));
[phi,lambda,~] = eigs(G_cen,d,'largestreal'); 
lambda = diag(lambda); 
Xi = sqrt(lambda') .* phi;  

end

