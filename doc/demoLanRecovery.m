clearvars, close all, write_fig = 0;
N = 5000;
ngrid_lan = 2000;
ngrid_cheb = 1000;
ngrid_sceig = 2000; 
ngrid_fitc = 750; 
rng(1)

% Choose data points
X = randn(N, 1);
%X = linspace(0,4,N)';

% Setup SKI and FITC
xg_lan = covGrid('create',X,true,ngrid_lan);
xg_cheb = covGrid('create',X,true,ngrid_cheb);
xg_sceig = covGrid('create',X,true,ngrid_sceig);
xu = linspace(min(X),max(X),ngrid_fitc)';

cov = {@(varargin)covMaterniso(3,varargin{:})};
%cov = {@covSEiso};
covg_lan = {@apxGrid,cov,xg_lan};
covg_cheb = {@apxGrid,cov,xg_cheb};
covg_sceig = {@apxGrid,cov,xg_sceig};
covf = {@apxSparse,cov,xu};
lik = {@likGauss};
means = {@meanZero};

% Generate data
hyp = struct('mean', [], 'cov', log([0.01;0.5]), 'lik', log(0.05));
opt_Y.hyp = hyp; opt_Y.type = 'Matern'; opt_Y.cov = cov;
%Y = f(X) + .1*randn(N,1);

time = zeros(1,7);

% Build surrogate
opt_sur.npts = 300; opt_sur.ntrials = 1000; opt_sur.param = {{'cov','lik'},[2,1]};
opt_sur.bounds = log([5e-3,0.1,1e-2;0.1,1,0.1]); opt_sur.method = 'lanczos';
opt_sur.cg_maxit = 1000; opt_sur.cg_tol = 1e-5; opt_sur.replace_diag = 0; opt_sur.nZ = 10;
opt_sur.kmax = 100; opt_sur.reorth = 1;

% Lan + Diag_Corr
opt2.cg_maxit = 1000; opt2.cg_tol = 1e-3; opt2.replace_diag = 0;
opt2.ldB2_lan = 1; opt2.ldB2_lan_reorth = 1; opt2.ldB2_lan_kmax = 100;

% Chebyshev
opt3.cg_maxit = 1000; opt3.cg_tol = 1e-3; opt3.replace_diag = 0; opt3.ldB2_cheby = 1;
inf3 = @(varargin)infGaussLik(varargin{:},opt3);
opt3.ldB2_cheby_degree = 200;

% Apx + No_Diag_Corr
opt4.cg_maxit = 2000; opt4.cg_tol = 1e-3; opt4.replace_diag = 0;
inf4 = @(varargin)infGaussLik(varargin{:},opt4);

% FITC
opt5.cg_maxit = 1000; opt5.cg_tol = 1e-3;
inf5 = @(varargin)infGaussLik(varargin{:},opt5);

% Exact
opt6.cg_maxit = 1000; opt6.cg_tol = 1e-3;
inf6 = @(varargin)infGaussLik(varargin{:},opt6);

for nrun = 1:1
    hyp0 = struct('mean', [], 'cov', 0.95*hyp.cov,'lik', 0.95*hyp.lik);
    Y = generate_data(X,opt_Y);
    data(:,nrun) = Y;
    hyp_recover = cell(6,1);
    NLK = cell(6,1);
    
    % Check derivatives
    K = apx(hyp,covg_lan,X,opt2);[ldB2,solveKiW,dW,dldB2,L] = K.fun(ones(N,1)/exp(hyp.lik*2));temp=dldB2(zeros(N,1));
    fprintf('Lanczos: (%.3e, %.3e, %.3e)\n',[ldB2,temp.cov'])
    
    K = apx(hyp,covg_sceig,X,opt3);[ldB2,solveKiW,dW,dldB2,L] = K.fun(ones(N,1)/exp(hyp.lik*2));temp=dldB2(zeros(N,1));
    fprintf('Cheb: (%.3e, %.3e, %.3e)\n',[ldB2,temp.cov'])
    
    K = apx(hyp,covg_sceig,X,opt4);[ldB2,solveKiW,dW,dldB2,L] = K.fun(ones(N,1)/exp(hyp.lik*2));temp=dldB2(zeros(N,1));
    fprintf('Scaled eig: (%.3e, %.3e, %.3e)\n',[ldB2,temp.cov'])
    
    K = apx(hyp,covf,X,opt5);[ldB2,solveKiW,dW,dldB2,L] = K.fun(ones(N,1)/exp(hyp.lik*2));temp=dldB2(zeros(N,1));
    fprintf('FITC: (%.3e, %.3e, %.3e)\n',[ldB2,temp.cov'])
    
    K = apx(hyp,cov,X,opt6);[ldB2,solveKiW,dW,dldB2,L] = K.fun(ones(N,1)/exp(hyp.lik*2));temp=dldB2(zeros(N,1));
    fprintf('Exact: (%.3e, %.3e, %.3e)\n',[ldB2,temp.cov'])
    
    %{
    for j = 1:1
        sur = build_surrogate(covg_lan,X,opt_sur);
        % Sur + Lan + Diag_Corr
        opt1.cg_maxit = 1000; opt1.cg_tol = 1e-3; opt1.replace_diag = 1;
        opt1.ldB2_sur = sur;
        inf1 = @(varargin)infGaussLik(varargin{:},opt1);
        % Sur + Lan + Diag_Corr
        tic;
        temp1 = minimize(hyp0,@gp,-30,inf1,means,covg_lan,lik,X,Y);
        hyp_sur(j,:) = exp([temp1.cov',temp1.lik]);
        time(1) = time(1) + toc;
        [~,nlZ,dnlZ] = infGaussLik(temp1,means,cov,lik,X,Y,opt6);
        NLK{1}(j,:) = [nlZ,dnlZ.cov',dnlZ.lik];
        fprintf('Surrogate: (%.3e, %.3e, %.3e)\n\n', exp([temp1.cov',temp1.lik]))
    end
    hyp_recover(1) ={hyp_sur};
    %}

    
    for j = 1:5
        % Lan + Diag_Corr
        opt2.ldB2_lan_hutch = sign(randn(N,10));
        inf2 = @(varargin)infGaussLik(varargin{:},opt2);
        tic;
        temp2 = minimize(hyp0,@gp,-30,inf2,means,covg_lan,lik,X,Y);
        hyp_lan(j,:) = exp([temp2.cov',temp2.lik]);
        time(2) = time(2) + toc;
        [~,nlZ,dnlZ] = infGaussLik(temp2,means,cov,lik,X,Y,opt6);
        NLK{2}(j,:) = [nlZ,dnlZ.cov',dnlZ.lik];
        fprintf('Lanczos: (%.3e, %.3e, %.3e)\n\n', exp([temp2.cov',temp2.lik]))
    end
    hyp_recover(2) ={hyp_lan};
    
    for j = 1:5
        % Chebyshev
        opt3.ldB2_cheby_hutch = sign(randn(N,10));
        inf3 = @(varargin)infGaussLik(varargin{:},opt3);
        tic;
        temp3 = minimize(hyp0,@gp,-30,inf3,means,covg_cheb,lik,X,Y);
        hyp_cheb(j,:) = exp([temp3.cov',temp3.lik]);
        time(3) = time(3) + toc;
        [~,nlZ,dnlZ] = infGaussLik(temp3,means,cov,lik,X,Y,opt6);
        NLK{3}(j, :) = [nlZ,dnlZ.cov',dnlZ.lik];
        fprintf('Cheb: (%.3e, %.3e, %.3e)\n\n', exp([temp3.cov',temp3.lik]))
    end
    hyp_recover(3) = {hyp_cheb};

    %{
    tic;
    temp4 = minimize(hyp0,@gp,-30,inf4,means,covg_sceig,lik,X,Y);
    hyp_recover(4) = {exp([temp4.cov',temp4.lik])};
    time(4) = time(4) + toc;
    [~,nlZ,dnlZ] = infGaussLik(temp4,means,cov,lik,X,Y,opt6);
    NLK(4) = {[nlZ,dnlZ.cov',dnlZ.lik]};
    fprintf('Scaled eig: (%.3e, %.3e, %.3e)\n\n', exp([temp4.cov',temp4.lik]))
    
    tic;
    temp5 = minimize(hyp0,@gp,-30,inf5,means,covf,lik,X,Y);
    hyp_recover(5) = {exp([temp5.cov',temp5.lik])};
    time(5) = time(5) + toc;
    [~,nlZ,dnlZ] = infGaussLik(temp5,means,cov,lik,X,Y,opt6);
    NLK(5) = {[nlZ,dnlZ.cov',dnlZ.lik]};
    fprintf('FITC: (%.3e, %.3e, %.3e)\n\n', exp([temp5.cov',temp5.lik]))
    
    tic;
    temp6 = minimize(hyp0,@gp,-30,inf6,means,cov,lik,X,Y);
    hyp_recover(6) = {exp([temp6.cov',temp6.lik])};
    time(6) = time(6) + toc;
    [~,nlZ,dnlZ] = infGaussLik(temp6,means,cov,lik,X,Y,opt6);
    NLK(6) = {[nlZ,dnlZ.cov',dnlZ.lik]};
    fprintf('Exact: (%.3e, %.3e, %.3e)\n\n', exp([temp6.cov',temp6.lik]))
    
    result{nrun} = hyp_recover;
    result_NLK{nrun} = NLK;
    %}
end

[~,nlZ,dnlZ] = infGaussLik(hyp,means,cov,lik,X,Y,opt6);
NLK(7) = {[nlZ,dnlZ.cov',dnlZ.lik]};
 
fprintf('\n\n==========================================\n')
fprintf('Surrogate: %.3e (%.3e, %.3e, %.3e) in %.3f (s)\n',NLK{1}(1), hyp_recover{1}, time(1))
if numel(hyp_recover{2}) == 3
    fprintf('Lanczos: %.3e (%.3e, %.3e, %.3e) in %.3f (s)\n',NLK{2}(1), hyp_recover{2}, time(2))
else
    nnruns = size(hyp_recover{2},1);
    fprintf('Lanczos: %.3e (%.3e, %.3e, %.3e)  in %.3f (s)\n',mean(NLK{2}(:,1)), exp(mean(log(hyp_recover{2}))),time(2)/nnruns)
end
if numel(hyp_recover{3}) == 3
    fprintf('Chebyshev: %.3e (%.3e, %.3e, %.3e) in %.3f (s)\n',NLK{3}(1), hyp_recover{3}, time(3))
else
    nnruns = size(hyp_recover{3},1);
    fprintf('Chebyshev: %.3e (%.3e, %.3e, %.3e)  in %.3f (s)\n',mean(NLK{3}(:,1)), exp(mean(log(hyp_recover{3}))),time(3)/nnruns)
end
fprintf('Scaled eig: %.3e (%.3e, %.3e, %.3e) in %.3f (s)\n',NLK{4}(1),hyp_recover{4}, time(4))
fprintf('FITC: %.3e (%.3e, %.3e, %.3e) in %.3f (s)\n',NLK{5}(1),hyp_recover{5}, time(5))
fprintf('Exact: %.3e (%.3e, %.3e, %.3e) in %.3f (s)\n',NLK{6}(1),hyp_recover{6}, time(6))
fprintf('True: %.3e (%.3e, %.3e, %.3e)\n', NLK{7}(1),exp([hyp.cov',hyp.lik]))