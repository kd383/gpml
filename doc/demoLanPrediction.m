clearvars, close all, write_fig = 0;

N = 1000;
X = 20*rand(N, 1) - 10;

% Set up dummy grid
xg = cell(1,1);
xg{1} = [-10,-5,5:10]';
m = length(xg{1});
xu = xg{1};
xs = linspace(-10, 10, 500)';

%cov = {@(varargin)covMaterniso(3,varargin{:})};
cov = {@covSEiso};
covg = {@apxGrid,cov,xg};
covf = {@apxSparse,cov,xu};
lik = {@likGauss};

% Generate data
hyp = struct('mean', [], 'cov', log([0.1;1]), 'lik', log(0.05));
%K = cov{:}(hyp.cov, X);
%K = 0.5*(K + K') + exp(2*hyp.lik)*eye(N);
%f = chol(K)'* randn(N,1);
a = 0.5; 
b = 1; 
means = {@meanSum,{@meanLinear,@meanConst}}; hyp.mean = [a;b];
f = @(x) a*x + b + sin(x);
fX = f(X) + 0.05*randn(N,1);

tick_size = 50;
line_width = 4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXACT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opt1.cg_maxit = 1000; 
opt1.cg_tol = 1e-3;

inf = @(varargin)infGaussLik(varargin{:},opt1);
hyp = minimize(hyp,@gp,-N,inf,means,cov,lik,X,fX);      % optimise hyperparameters
[ymu1,ys21] = gp(hyp,inf,means,cov,lik,X,fX,xs);
%{
subplot(3,2,1)
plot(xs, ymu1,'k','LineWidth',2), 
hold on
plot(X, fX, 'k.', 'MarkerSize', 15)
plot(xs, ymu1 + 2*sqrt(ys21),'r', 'LineWidth', 2)
plot(xs, ymu1 - 2*sqrt(ys21),'r', 'LineWidth', 2)
title('Exact', 'fontsize', 30)
set(gca,'fontsize',24)
%}
%%%%%%%%%%%%%%%%%%%%%%%%% LANCZOS (diag) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opt = [];
opt.cg_maxit = 1000;
opt.cg_tol = 1e-3;
opt.replace_diag = 1;
opt.ldB2_lan = 1;
opt.ldB2_lan_reorth = 1;
opt.ldB2_lan_kmax = 300;
opt.ldB2_lan_hutch = sign(randn(N,10));
inf = @(varargin)infGaussLik(varargin{:}, opt);

[ymu,ys2] = gp(hyp,inf,means,covg,lik,X,fX,xs);

figure('units','normalized','outerposition',[0 0 1 1])
hold off
plot(xs, ymu,'b','LineWidth',line_width), 
hold on
plot(X, fX, 'k.', 'MarkerSize', 10)
plot(xs, ymu + 2*sqrt(ys2),'r--', 'LineWidth', line_width)
plot(xs, ymu - 2*sqrt(ys2),'r--', 'LineWidth', line_width)
plot(xg{1}, f(xg{1}), 'gx', 'MarkerSize', 30, 'LineWidth', 2*line_width)
set(gca,'fontsize',tick_size)
set(gcf,'paperpositionmode','auto')
print(gcf,'-depsc2','-loose','../../Latex/icml/pics/pred_lanc_diag.eps');

    
%%%%%%%%%%%%%%%%%%%%%%%%% LANCZOS (no diag) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opt = [];
opt.cg_maxit = 1000;
opt.cg_tol = 1e-3;
opt.replace_diag = 0;
opt.ldB2_lan = 1;
opt.ldB2_lan_reorth = 1;
opt.ldB2_lan_kmax = 300;
opt.ldB2_lan_hutch = sign(randn(N,10));
inf = @(varargin)infGaussLik(varargin{:}, opt);

[ymu,ys2] = gp(hyp,inf,means,covg,lik,X,fX,xs);

figure('units','normalized','outerposition',[0 0 1 1])
plot(xs, ymu,'b','LineWidth',2), 
hold on
plot(X, fX, 'k.', 'MarkerSize', 10)
plot(xs, ymu + 2*sqrt(ys2),'r--', 'LineWidth', line_width)
plot(xs, ymu - 2*sqrt(ys2),'r--', 'LineWidth', line_width)
plot(xg{1}, f(xg{1}), 'gx', 'MarkerSize', 30, 'LineWidth', 2*line_width)
set(gca,'fontsize',tick_size)
set(gcf,'paperpositionmode','auto')
print(gcf,'-depsc2','-loose','../../Latex/icml/pics/pred_lanc_nodiag.eps');

%%%%%%%%%%%%%%%%%%%%%%%% Chebyshev (diag) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opt = [];
opt.cg_maxit = 1000;
opt.cg_tol = 1e-3;
opt.replace_diag = 1;
opt.ldB2_cheby = 1;
opt.ldB2_cheby_degree = 300;
opt.ldB2_cheby_hutch = sign(randn(N,10));
inf = @(varargin)infGaussLik(varargin{:}, opt);

[ymu,ys23] = gp(hyp,inf,means,covg,lik,X,fX,xs);

figure('units','normalized','outerposition',[0 0 1 1])
plot(xs, ymu,'b','LineWidth',2), 
hold on
plot(X, fX, 'k.', 'MarkerSize', 10)
plot(xs, ymu + 2*sqrt(ys23),'r--', 'LineWidth', line_width)
plot(xs, ymu - 2*sqrt(ys23),'r--', 'LineWidth', line_width)
plot(xg{1}, f(xg{1}), 'gx', 'MarkerSize', 30, 'LineWidth', 2*line_width)
set(gca,'fontsize',tick_size)
set(gcf,'paperpositionmode','auto')
print(gcf,'-depsc2','-loose','../../Latex/icml/pics/pred_cheb_diag.eps');

%%%%%%%%%%%%%%%%%%%%%%%% CHEBys2HEV (no diag) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opt = [];
opt.cg_maxit = 1000;
opt.cg_tol = 1e-3;
opt.replace_diag = 0;
opt.ldB2_cheby = 1;
opt.ldB2_cheby_degree = 300;
opt.ldB2_cheby_hutch = sign(randn(N,10));
inf = @(varargin)infGaussLik(varargin{:}, opt);

[ymu,ys23] = gp(hyp,inf,means,covg,lik,X,fX,xs);

figure('units','normalized','outerposition',[0 0 1 1])
plot(xs, ymu,'b','LineWidth',2), 
hold on
plot(X, fX, 'k.', 'MarkerSize', 10)
plot(xs, ymu + 2*sqrt(ys23),'r--', 'LineWidth', line_width)
plot(xs, ymu - 2*sqrt(ys23),'r--', 'LineWidth', line_width)
plot(xg{1}, f(xg{1}), 'gx', 'MarkerSize', 30, 'LineWidth', 2*line_width)
set(gca,'fontsize',tick_size)
set(gcf,'paperpositionmode','auto')
print(gcf,'-depsc2','-loose','../../Latex/icml/pics/pred_cheb_nodiag.eps');

%%%%%%%%%%%%%%%%%%%%%%%%%% FITC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opt = [];
opt.cg_maxit = 1000;
opt.cg_tol = 1e-3;
opt.replace_diag = 0;
inf = @(varargin)infGaussLik(varargin{:}, opt);
[ymu,ys25] = gp(hyp,inf,means,covf,lik,X,fX,xs);

figure('units','normalized','outerposition',[0 0 1 1])
plot(xs, ymu,'b','LineWidth',2), 
hold on
plot(X, fX, 'k.', 'MarkerSize', 10)
plot(xs, ymu + 2*sqrt(ys25),'r--', 'LineWidth', line_width)
plot(xs, ymu - 2*sqrt(ys25),'r--', 'LineWidth', line_width)
plot(xg{1}, f(xg{1}), 'gx', 'MarkerSize', 30, 'LineWidth', 2*line_width)
set(gca,'fontsize',tick_size)
set(gcf,'paperpositionmode','auto')
print(gcf,'-depsc2','-loose','../../Latex/icml/pics/pred_fitc.eps');

%%%%%%%%%%%%%%%%%%%%%%%%%% Scaled eig %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opt = [];
opt.cg_maxit = 1000;
opt.cg_tol = 1e-3;
opt.replace_diag = 0;
inf = @(varargin)infGaussLik(varargin{:}, opt);
[ymu,ys24] = gp(hyp,inf,means,covg,lik,X,fX,xs);

figure('units','normalized','outerposition',[0 0 1 1])
plot(xs, ymu,'b','LineWidth',2), 
hold on
plot(X, fX, 'k.', 'MarkerSize', 10)
plot(xs, ymu + 2*sqrt(ys24),'r--', 'LineWidth', line_width)
plot(xs, ymu - 2*sqrt(ys24),'r--', 'LineWidth', line_width)
plot(xg{1}, f(xg{1}), 'gx', 'MarkerSize', 30, 'LineWidth', 2*line_width)
set(gca,'fontsize',tick_size)
set(gcf,'paperpositionmode','auto')
print(gcf,'-depsc2','-loose','../../Latex/icml/pics/pred_sceig.eps');