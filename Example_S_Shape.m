%% Synthetic example of the S shape

% Data generation
n = 200;
rng(33)

Z_1 = random('Uniform',-3*pi/2,3*pi/2,[n,1]);
Z_1 = sort(Z_1);
Z_2 = random('Uniform',1,4,[n,1]);
X = [sin(Z_1) Z_2 sign(Z_1).*(cos(Z_1)-1)];

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

fname = sprintf('S_Shape_n200_results');
save(fname,'X','Xi_all','Xi_ave','W','K_Iso','K_LLE','P_tSNE','K_UMAP','K_PTU');
    
% Figures
col_seq = 1:n;

figure
scatter3(X(:,1),X(:,2),X(:,3),[],col_seq)
view([-0.5 -1.6 0.5])
set(gca,'XColor', 'none','YColor','none','ZColor','none') 
print(gcf,'S-Shape_n1000','-dpng'); 

figure
scatter(Xi_all{1,1}(:,1),Xi_all{1,1}(:,2),[],col_seq)
xlim([-5.5 5.5])
ylim([-1.8 2])
set(gca,'FontSize',20)
text(2.7,-1.5,'Isomap','FontSize',24) 
print(gcf,'S_Shape_Isomap','-dpng'); 

figure
scatter(Xi_all{1,2}(:,1),Xi_all{1,2}(:,2),[],col_seq)
xlim([-0.12 0.12])
ylim([-0.17 0.15])
set(gca,'FontSize',20)
text(0.085,-0.15,'LLE','FontSize',24) 
print(gcf,'S_Shape_LLE','-dpng'); 

figure
scatter(Xi_all{1,3}(:,1),Xi_all{1,3}(:,2),[],col_seq)
xlim([-4.5 4.5])
ylim([-4.2 4.2])
set(gca,'FontSize',20)
text(3,-3.7,'tSNE','FontSize',24) 
print(gcf,'S_Shape_tSNE','-dpng'); 

figure
scatter(Xi_all{1,4}(:,1),Xi_all{1,4}(:,2),[],col_seq)
xlim([1 25])
ylim([-4 10])
set(gca,'FontSize',20)
text(20,-3,'UMAP','FontSize',24) 
print(gcf,'S_Shape_UMAP','-dpng'); 

figure
scatter(Xi_all{1,5}(:,1),Xi_all{1,5}(:,2),[],col_seq)
xlim([-5 5])
ylim([-2 2])
set(gca,'FontSize',20)
text(3.5,-1.8,'PTU','FontSize',24) 
print(gcf,'S_Shape_PTU','-dpng'); 

figure
scatter(Xi_ave(:,1),Xi_ave(:,2),[],col_seq)
xlim([-2 2])
ylim([-1.6 1.6])
set(gca,'FontSize',20)
text(1.1,-1.4,'MAML','FontSize',24) 
print(gcf,'S_Shape_MAML','-dpng'); 
