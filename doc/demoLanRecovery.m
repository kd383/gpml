clear all, close all, write_fig = 0; 
N = 1000;
X = sort(4*rand(N,1)-2,'ascend');
%Y = sin(X);
xg = apxGrid('create',X,true,100);
cov = {@apxGrid,{@covSEiso},xg};
lik = {@likGauss};
means = {@meanZero};
hyp = struct('mean', [], 'cov', log([0.1,0.8]), 'lik', log(0.02));
hyp0 = struct('mean', [], 'cov', log([0.2,1]), 'lik', log(0.1));

time = zeros(1,3);
opt.npts = 200; opt.ntrials = 1000; opt.param = {{'cov','lik'},[2,1]};
opt.bounds = log([5e-2,0.1,1e-2;1,3,1]); opt.method = 'lanczos';
opt.cg_maxit = 500; opt.cg_tol = 1e-5;

sur = build_surrogate(cov,X,opt);

opt1.cg_maxit = 500; opt1.cg_tol = 1e-5;
opt1.ldB2_sur = sur;
inf1 = @(varargin)infGaussLik(varargin{:},opt1);
opt2.cg_maxit = 500; opt2.cg_tol = 1e-5;
inf = @(varargin)infGaussLik(varargin{:},opt2);


for nrun = 1:5
    % Generate data
    K = covSEiso(hyp.cov,X);
    K = (K+K')/2+exp(2*hyp.lik)*eye(N);
    Y = feval(means{:},hyp.mean,X) + chol(K)'*randn(N,1);
    
    % Surrogate + Lancozs + ApxGrid
    tic;
    temp1 = minimize(hyp0,@gp,-200,inf1,means,cov,lik,X,Y);
    hyp_est(nrun,:) = exp([temp1.cov,temp1.lik]);
    % hyp_est(nrun,:) = exp([temp1.cov,temp1.lik,temp1.mean]);
    time(2) = time(2) + toc;
    
    % ApxGrid
    
    tic;
    temp2 = minimize(hyp0,@gp,-200,inf,means,cov,lik,X,Y);
    hyp_exact(nrun,:) = exp([temp2.cov,temp2.lik]);
    % hyp_exact(nrun,:) = exp([temp2.cov,temp2.lik,temp2.mean]);
    time(3) = time(3) + toc;
    
end

result = exp([hyp.cov';hyp.lik]);
% result = [exp([hyp.cov';hyp.lik]);hyp.mean'];
%[result hyp_est' hyp_true'] % Use this line when only one run
result = [result, mean(hyp_est)', std(hyp_est)', mean(hyp_exact)', std(hyp_exact)'];
T = array2table(result,'VariableNames',{'True','Sur_Lan_Grid','Error1','Grid','Error2'})
time(2:3) = time(2:3)/5;
fprintf('Time for Sur+Lan+Grid optimization is %.3f,\nfor Grid optimization is %.3f\n',time(2:3));
