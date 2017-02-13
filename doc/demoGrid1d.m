disp('See http://www.gaussianprocess.org/gpml/code/matlab/doc/ for details.')
clear all, close all, write_fig = 0; N = 30;
sd = 3; rand('seed',sd), randn('seed',sd)       % set a seed for reproducability
liktyp = input('Which likelihood?\n  (g)aussian, (l)ogistic: ','s');

a = 0.3; b = 1.2; f = @(x) a*x + b + sin(x);               % underlying function
n = 30; sn = 0.5;          % number of training points, noise standard deviation
x = 2*rand(n,1)-1; x = 1+4*x+sign(x); y = f(x)+sn*randn(n,1);      % sample data

cov = {@covSEiso}; sf = 2; ell = 1.0; hyp.cov = log([ell;sf]);
opt = [];
%{
opt_sur.npts = 200; opt_sur.ntrials = 1000; opt_sur.param = {{'cov','lik'},[2,1]};
opt_sur.bounds = log([5e-1,1,1e-2;2,3,1]); opt_sur.method = 'lanczos';
opt_sur.cg_maxit = 500; opt_sur.cg_tol = 1e-5;

sur = build_surrogate(cov,x,opt_sur);

opt1.cg_maxit = 500; opt1.cg_tol = 1e-5;
opt1.ldB2_sur = sur;
inf1 = @(varargin)infGaussLik(varargin{:},opt1);
%}
mean = {@meanSum,{@meanLinear,@meanConst}}; hyp.mean = [a;b];
if isequal(liktyp,'g')
  lik = {@likGauss};    hyp.lik = log(sn); inf = @(varargin)infGaussLik(varargin{:},opt);
else
  lik = {@likLogistic}; hyp.lik = [];      inf = @infLaplace;  y = sign(y);
end

fprintf('Optimise hyperparameters.\n')
hyp = minimize(hyp,@gp,-N,inf,mean,cov,lik,x,y);      % optimise hyperparameters
xs = linspace(-8,10,2e3)'; ys = f(xs);                   % exact function values

[ymu,ys2] = gp(hyp,inf,mean,cov,lik,x,y,xs);                  % dense prediction
[nlZ,dnlZ] = gp(hyp,inf,mean,cov,lik,x,y); % marginal likelihood and derivatives

nu = 10; xu = linspace(-6,8,nu)'; covf = {@apxSparse,cov,xu};  % FITC prediction
[ymuf,ys2f] = gp(hyp,inf,mean,covf,lik,x,y,xs);

ng = 40; xg = linspace(-6,8,ng)'; covg = {@apxGrid,{cov},{xg}};% grid prediction
opt.cg_maxit = 500; opt.cg_tol = 1e-5; opt.pred_var = 100;          % parameters
inf = @(varargin) infGrid(varargin{:},opt);
[ymug,ys2g] = gp(hyp,inf,mean,covg,lik,x,y,xs);
[postg,nlZg,dnlZg] = infGrid(hyp,mean,covg,lik,x,y,opt);  % fast grid prediction
[fmugf,fs2gf,ymugf,ys2gf] = postg.predict(xs);

fprintf('Sampling estimators for nlZ and dnlZ\n')
opt.ndcovs = 25;       % ask for (additional) sampling-based (exact) derivatives
opt.ldB2_lan = true;
opt.ldB2_lan_hutch = 10;
[posts,nlZs,dnlZs] = infGrid(hyp,mean,covg,lik,x,y,opt);
[fmugl,fs2gl,ymugl,ys2gl] = posts.predict(xs);
[dnlZ.cov,dnlZg.cov,dnlZs.covs,dnlZs.cov]
[nlZ,nlZg,nlZs]

subplot(211)
plot(xs,ymu,'k','LineWidth',2), hold on
plot(xs,ymuf,'g-.','LineWidth',2)
plot(xs,ymug,'m:','LineWidth',2)
plot(xs,ymugf,'c--','LineWidth',2)
plot(xs,ymugl,'y.','LineWidth',2)
legend('exact','FITC','grid','fast-grid','lan'), title('Predictive mean')
plot(x,y,'r+'), plot(xs,ys,'r')
plot(xs,ymu+2*sqrt(ys2),'k'), plot(xs,ymu-2*sqrt(ys2),'k')
xlim([-8,10]), ylim([-3,6])

subplot(212)
plot(xs,sqrt(ys2),'k','LineWidth',2), hold on
plot(xs,sqrt(ys2f),'g-.','LineWidth',2)
plot(xs,sqrt(ys2g),'m:','LineWidth',2)
plot(xs,sqrt(ys2gf),'c--','LineWidth',2)
plot(xs,sqrt(ys2gl),'y.','LineWidth',2)
legend('exact','FITC','grid','fast-grid','lan'), title('Predictive standard dev')
xlim([-8,10]), if write_fig, print -depsc f10.eps; end