%% Synthetic example of the Swiss roll

% Data generation
n = 1000;
Z_1 = linspace(3,10,n)';
rng(33)
Z_2 = random('Uniform',0,3,[n,1]);
X = [Z_1.*cos(Z_1) Z_1.*sin(Z_1) Z_2]; 
col_seq = 1:n; 

% Apply manifold learning algorithms with tuning parameter chosen by the
% proposed metric
d = 2;
n_can = 10;

% Isomap
K = round(linspace(10,40,n_can));
S_can = zeros(n_can,1);
for i = 1:n_can
     Xi_Iso  = Isomap( X,K(i),d );
     S_can(i) = AUC_R(X,Xi_Iso); 
end
[~,ind] = max(S_can);
K_Iso = K(ind);
Xi_Iso = Isomap( X,K_Iso,d );

% LLE
K = round(linspace(10,40,n_can));
S_can = zeros(n_can,1);
for i = 1:n_can
    Xi_LLE = LLE( X,K(i),d);
    S_can(i) = AUC_R(X,Xi_LLE); 
end
[~,ind] = max(S_can);
K_LLE = K(ind);
Xi_LLE = LLE( X,K_LLE,d);
     
% t-SNE
P = round(linspace(30,100,n_can));
S_can = zeros(n_can,1);
for i = 1:n_can
     Xi_tSNE  = tsne(X,'Algorithm','Exact','NumDimensions',d,'Perplexity',P(i));
     S_can(i) = AUC_R(X,Xi_tSNE);
end
[~,ind] = max(S_can);
P_tSNE = P(ind);
Xi_tSNE  = tsne(X,'Algorithm','Exact','NumDimensions',d,'Perplexity',P_tSNE);
     
% UMAP 
K = round(linspace(10,40,n_can));
S_can = zeros(n_can,1);
for i = 1:n_can
     Xi_UMAP  = run_umap(X,'n_neighbors',K(i),'n_components',d,'min_dist',1,'verbose','none');
     S_can(i) = AUC_R(X,Xi_UMAP);
end
[~,ind] = max(S_can);
K_UMAP = K(ind);
Xi_UMAP  = run_umap(X,'n_neighbors',K_UMAP,'n_components',d,'min_dist',1,'verbose','none');

% PTU
K = round(linspace(10,30,n_can)); 
S_can = zeros(n_can,1);
for i = 1:n_can
    Xi_PTU = PTU( X,K(i),K(i),d,0 );
    S_can(i) = AUC_R(X,Xi_PTU);  
end
[~,ind] = max(S_can);
K_PTU = K(ind);
Xi_PTU = PTU( X,K_PTU,K_PTU,d,0 );

% Model averaging
m = 5;
Xi_all = cell(1,m);
Xi_all{1,1} = Xi_Iso;
Xi_all{1,2} = Xi_LLE;
Xi_all{1,3} = Xi_tSNE;
Xi_all{1,4} = Xi_UMAP;
Xi_all{1,5} = Xi_PTU;
[Xi_ave,W] = MMA(X,Xi_all);

fname = sprintf('SwissRoll_results');
save(fname,'X','Xi_all','Xi_ave','W','K_Iso','K_LLE','P_tSNE','K_UMAP','K_PTU');
    
% Figures
figure
scatter3(X(:,1),X(:,2),X(:,3),[],col_seq)
view([-0.5 -0.75 0.75])
set(gca,'XColor', 'none','YColor','none','ZColor','none') 
print(gcf,'SwissRoll','-dpng'); 
 
figure
scatter(Xi_all{1,1}(:,1),Xi_all{1,1}(:,2),[],col_seq)
xlim([-28 20])
ylim([-2.2 2.2])
set(gca,'FontSize',20)
text(9,-1.8,'Isomap','FontSize',24) 
print(gcf,'SwissRoll_Isomap','-dpng'); 

figure
scatter(Xi_all{1,2}(:,1),Xi_all{1,2}(:,2),[],col_seq)
xlim([-0.07 0.05]) 
ylim([-0.11 0.06])
set(gca,'FontSize',20)
text(0.035,-0.1,'LLE','FontSize',24) 
print(gcf,'SwissRoll_LLE','-dpng'); 

figure
scatter(Xi_all{1,3}(:,1),Xi_all{1,3}(:,2),[],col_seq) 
ylim([-35 50])
set(gca,'FontSize',20)
text(30,-30,'tSNE','FontSize',24) 
print(gcf,'SwissRoll_tSNE','-dpng'); 

figure
scatter(Xi_all{1,4}(:,1),Xi_all{1,4}(:,2),[],col_seq)  
xlim([-10 15])
set(gca,'FontSize',20)
text(10,-37,'UMAP','FontSize',24) 
print(gcf,'SwissRoll_UMAP','-dpng'); 

figure
scatter(Xi_all{1,5}(:,1),Xi_all{1,5}(:,2),[],col_seq)  
xlim([-28 20])
ylim([-2.5 2.5])
set(gca,'FontSize',20)
text(13,-2.2,'PTU','FontSize',24) 
print(gcf,'SwissRoll_PTU','-dpng');

figure
scatter(Xi_ave(:,1),Xi_ave(:,2),[],col_seq)  
xlim([-2.2 1.6])
ylim([-2.1 2.1])
set(gca,'FontSize',20)
text(0.8,-1.9,'MAML','FontSize',24) 
print(gcf,'SwissRoll_MAML','-dpng');

