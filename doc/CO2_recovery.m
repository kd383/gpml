clearvars -except sur, close all, write_fig = 0;
load 'CO2.mat'
X = CO2(:,1:2);
xg = {{(-82:82)'},{(-143.5:143.5)'}};
cov = {{@covSEiso},{@covSEiso}};
covg = {@apxGrid,cov,xg};
means = {@meanZero}; lik = {@likGauss};
opt.cg_maxit = 500; opt.cg_tol = 1e-5;
inf = @(varargin) infGrid(varargin{:},opt);

opt_sur.npts = 75; opt_sur.ntrials = 1000; opt_sur.param = {{'cov','lik'},[4,1]};
opt_sur.bounds = log([5,0.5,5,0.5,1e-2;15,2,15,2,1]); opt_sur.method = 'lanczos';
opt_sur.cg_maxit = 500; opt_sur.cg_tol = 1e-5;
tic;
sur = build_surrogate(covg,CO2(:,1:2),opt_sur);
time = toc

opt1.cg_maxit = 700; opt1.cg_tol = 1e-3;
opt1.ldB2_sur = sur; opt1.replace_diag = 0;
inf1 = @(varargin)infGrid(varargin{:},opt1);

hyp0 = struct('mean', [], 'cov', log([8;1;8;1]), 'lik', log(0.1));
tic;
hyp = minimize(hyp0,@gp,-100,inf1,means,covg,lik,CO2(:,1:2),CO2(:,3));
toc
opt_pre.cg_maxit = 500; opt_pre.cg_tol = 1e-5;
opt_pre.ldB2_lan = 1;
[post,nlZ,dnlZ] = infGrid(hyp,means,covg,lik,CO2(:,1:2),CO2(:,3),opt_pre);
[xs,ns] = apxGrid('expand',xg);
[fmu,fs2,ymu,ys2] = post.predict(xs);