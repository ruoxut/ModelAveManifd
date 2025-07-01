%% Synthetic example of the 4-petal shape

% Data generation
n = 1000;
rng(33)
theta = random('Uniform',pi/4,pi,[n,1]);
theta(1:n/4) = sort(theta(1:n/4));theta(n/4+1:n/2) = sort(theta(n/4+1:n/2));
theta(n/2+1:3*n/4) = sort(theta(n/2+1:3*n/4));theta(3*n/4+1:n) = sort(theta(3*n/4+1:n));

phi = [random('Uniform',0,pi/3,[n/4,1]);
       random('Uniform',pi/2,5*pi/6,[n/4,1]);
       random('Uniform',pi,4*pi/3,[n/4,1]);
       random('Uniform',3*pi/2,11*pi/6,[n/4,1])];

X = [sin(theta).*cos(phi) sin(theta).*sin(phi) cos(theta)];
col_seq = 1:n/4;
col_seq = repmat(col_seq,1,4);

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

fname = sprintf('4Petal_results');
save(fname,'X','Xi_all','Xi_ave','W','K_Iso','K_LLE','P_tSNE','K_UMAP','K_PTU');
    
% Figures
figure
scatter3(X(:,1),X(:,2),X(:,3),[],col_seq)
theta_0 = linspace(0,pi,100);
phi_0 = linspace(-pi,pi,16);
%hold on
%for i = 1:length(phi_0)
%    plot3(sin(theta_0).*cos(phi_0(i)),sin(theta_0).*sin(phi_0(i)),cos(theta_0),'--','Color','blue')
%end
hold off
view([0.8 0.45 1.2])
set(gca,'XColor', 'none','YColor','none','ZColor','none') 
print(gcf,'4-petal','-dpng'); 

figure
scatter(Xi_all{1,1}(:,1),Xi_all{1,1}(:,2),[],col_seq)
xlim([-2 2])
ylim([-1.8 1.5])
set(gca,'FontSize',20)
text(1.1,-1.6,'Isomap','FontSize',24) 
print(gcf,'4Petal_Isomap','-dpng'); 

figure
scatter(Xi_all{1,2}(:,1),Xi_all{1,2}(:,2),[],col_seq)
xlim([-0.07 0.07])
ylim([-0.07 0.07])
set(gca,'FontSize',20)
text(0.05,-0.06,'LLE','FontSize',24) 
print(gcf,'4Petal_LLE','-dpng'); 

figure
scatter(Xi_all{1,3}(:,1),Xi_all{1,3}(:,2),[],col_seq)
xlim([-22 22])
ylim([-22 22])
set(gca,'FontSize',20)
text(14,-18,'tSNE','FontSize',24) 
print(gcf,'4Petal_tSNE','-dpng'); 

figure
scatter(Xi_all{1,4}(:,1),Xi_all{1,4}(:,2),[],col_seq)
xlim([-17 15]) 
ylim([-17 17])
set(gca,'FontSize',20)
text(8,-14,'UMAP','FontSize',24) 
print(gcf,'4Petal_UMAP','-dpng'); 

figure
scatter(Xi_all{1,5}(:,1),Xi_all{1,5}(:,2),[],col_seq)
xlim([-2.6 2.6]) 
ylim([-2.7 2.7])
set(gca,'FontSize',20)
text(1.8,-2.3,'PTU','FontSize',23) 
print(gcf,'4Petal_PTU','-dpng'); 

figure
scatter(Xi_ave(:,1),Xi_ave(:,2),[],col_seq)
xlim([-2.2 2.3])
ylim([-2.6 2.6])
set(gca,'FontSize',20)
text(1.4,-2.3,'MAML','FontSize',24) 
print(gcf,'4Petal_MAML','-dpng'); 
