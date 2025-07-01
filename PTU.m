function [ Xi,D,Path ] = PTU( X,K,K_pca,d,opt )
% Parallel transport unfolding.
% Input: 
% X: n*D data matrix;
% K: number of nearest neighbours;
% K_pca: number of nearest neighbours used in local PCA;
% d: intrinsic dimension;
% opt: =1 means rescale; otherwise not rescale;
% Output:
% Xi: n*d low-dim representations;
% D: n*n proximity graph.
% Path: n*n cell, (i,j)-cell contains the shortest path indices from i to j.

% Author: Ruoxu Tan; date: 2022/Oct/29; Matlab version: R2020a.

if K_pca < d
    error('K_pca is smaller than the intrinsic dimension.')
end 

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

% Shortest path graph and corresponding paths
G_s = zeros(n);
Path = cell(n);
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
    
    for j = 1:n
        if j == i 
            Path{i,j} = i;
        else 
            u = j;
            while u ~= 0            
            Path{i,j} = [u, Path{i,j}];
            u = Pre(u);
            end
        end
    end
end
 
%% Tangent spaces and parallel transport   
mu = cell(1,n);
TM = cell(1,n);% Tangent spaces
for i = 1:n
    [~,ind] = sort(G_s(i,:));
    mu{1,i} = mean(X(ind(2:K_pca+1),:),1);
    X_i_cen = X(ind(2:K_pca+1),:)-mu{1,i};  
    [~,~,phi_i] = svds(X_i_cen,d);
    TM{1,i} = phi_i;
end

% Discrete parallel transport
R = cell(n);
for i = 1:n
    for j = 1:n
        if j ~= i
            Phi_ij = TM{1,i}' * TM{1,j};
            [U,~,V] = svd(Phi_ij); 
            R{j,i} = V * U';
        end
    end
end

% Proximity graph based on parallel transport
D = Inf * ones(n);
for i = 1:n
    D(i,i) = 0;
    for j = 1:n
        if length(Path{i,j}) > 1
            v_i = zeros(d,length(Path{i,j})-1);
            for k = 1:size(v_i,2)    
                xi_i = TM{1,Path{i,j}(k+1)}' * (X(Path{i,j}(k),:)' - X(Path{i,j}(k+1),:)') ;
                if opt == 1
                xi_i = xi_i ./ norm(xi_i) .* norm(X(Path{i,j}(k),:)' - X(Path{i,j}(k+1),:)');% Rescaling
                end    
                s = k+1;
                while Path{i,j}(s) ~= j
                    xi_i = R{Path{i,j}(s+1),Path{i,j}(s)} * xi_i; %T_s M to T_s+1 M                  
                    s = s+1;
                end
                v_i(:,k) = xi_i;
            end
            v = sum(v_i,2);
            D(i,j) = norm(v);
        end
    end
end
D = (D+D')./2;

% Updated G_s using D
G_s = D; 

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

