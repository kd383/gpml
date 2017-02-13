clear all, close all, write_fig = 0; 
N = 5000;
X = sort(4*rand(N,1)-2,'ascend');
xg = apxGrid('create',X,true,500);
xu = linspace(-2,2,500)';
cov = {@covSEiso};
covg = {@apxGrid,cov,xg};
covf = {@apxSparse,cov,xu};
lik = {@likGauss};
means = {@meanZero};
%hyp = struct('mean', [], 'cov', log([0.1,0.8]), 'lik', log(0.1));
%hyp0 = struct('mean', [], 'cov', log([0.2,1]), 'lik', log(0.2));
ell_test = log([0.15,0.4]);
sf_test = log([0.5,1.2]);
sigma_test = log([0.1,0.4]);

time = zeros(1,5);
opt.npts = 150; opt.ntrials = 1000; opt.param = {{'cov','lik'},[2,1]};
opt.bounds = log([5e-2,0.1,1e-2;1,2,1]); opt.method = 'lanczos';
opt.cg_maxit = 500; opt.cg_tol = 1e-5;
tic;
sur = build_surrogate(covg,X,opt);
time(1) = toc

opt1.cg_maxit = 500; opt1.cg_tol = 1e-5;
opt1.ldB2_sur = sur; opt1.replace_diag = 0;
inf1 = @(varargin)infGaussLik(varargin{:},opt1);
opt2.cg_maxit = 500; opt2.cg_tol = 1e-5;
inf = @(varargin)infGaussLik(varargin{:},opt2);


for nrun = 1:5
    % Generate data
    hyp = struct('mean', [], 'cov', [ell_test(1) sf_test(1)], 'lik', sigma_test(1));
    K = covSEiso(hyp.cov,X);
    K = (K+K')/2+exp(2*hyp.lik)*eye(N);
    Y = feval(means{:},hyp.mean,X) + chol(K)'*randn(N,1);
    hyp0 = struct('mean', [], 'cov', 1.5*[ell_test(1),sf_test(1)],'lik', 1.5*sigma_test(1));
    
    % Surrogate + Lancozs + ApxGrid
    tic;
    temp1 = minimize(hyp0,@gp,-100,inf1,means,covg,lik,X,Y);
    hyp_est(nrun,:) = exp([temp1.cov,temp1.lik]);
    % hyp_est(nrun,:) = exp([temp1.cov,temp1.lik,temp1.mean]);
    time(2) = time(2) + toc;
    
    % ApxGrid
    
    tic;
    temp2 = minimize(hyp0,@gp,-100,inf,means,covg,lik,X,Y);
    hyp_grid(nrun,:) = exp([temp2.cov,temp2.lik]);
    % hyp_exact(nrun,:) = exp([temp2.cov,temp2.lik,temp2.mean]);
    time(3) = time(3) + toc;
    %{
    tic;
    temp3 = minimize(hyp0,@gp,-200,inf,means,cov,lik,X,Y);
    hyp_exact(nrun,:) = exp([temp3.cov,temp3.lik]);
    time(4) = time(4) + toc;
    %}
    tic;
    temp3 = minimize(hyp0,@gp,-100,inf,means,covf,lik,X,Y);
    hyp_sparse(nrun,:) = exp([temp3.cov,temp3.lik]);
    time(4) = time(4) + toc;
    
end

result = exp([hyp.cov';hyp.lik]);
% result = [exp([hyp.cov';hyp.lik]);hyp.mean'];
%[result hyp_est' hyp_exact'] % Use this line when only one run
result = [result, mean(hyp_est)', std(hyp_est)', mean(hyp_grid)', std(hyp_grid)',...
    mean(hyp_sparse)', std(hyp_sparse)'];
T = array2table(result,'VariableNames',{'True','Sur_Lan_Grid','Error1',...
    'Grid','Error2','Sparse','Error3'})
time(2:5) = time(2:5)/5;
method = {'Sur','Grid','Sparse'};
for k = 1:3
    fprintf('Time for %s optimization is %.3f.\n',method{k},time(k+1));
end
