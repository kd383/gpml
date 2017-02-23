clearvars, close all, write_fig = 0; 
N = 1000; ngrid = 3000;

% Choose data points
X = sort(4*rand(N,1)-2,'ascend');
%X = linspace(0,4,N)';

% Setup SKI and FITC
xg = covGrid('create',X,true,ngrid);
xu = linspace(-2,2,ngrid/3)';

cov = {@(varargin)covMaterniso(3,varargin{:})};
%cov = {@covSEiso};
covg = {@apxGrid,cov,xg};
covf = {@apxSparse,cov,xu};
lik = {@likGauss};
means = {@meanZero};

% Generate data
hyp = struct('mean', [], 'cov', log([0.01;0.5]), 'lik', log(0.05));
opt_Y.hyp = hyp; opt_Y.type = 'Matern'; opt_Y.cov = cov;
%Y = f(X) + .1*randn(N,1);

time = zeros(1,7);

% Build surrogate 
opt_sur.npts = 200; opt_sur.ntrials = 1000; opt_sur.param = {{'cov','lik'},[2,1]};
opt_sur.bounds = log([5e-3,0.1,1e-2;0.1,1,0.1]); opt_sur.method = 'lanczos';
opt_sur.cg_maxit = 1000; opt_sur.cg_tol = 1e-5; opt_sur.replace_diag = 1; opt_sur.nZ = 10;
opt_sur.kmax = 100; opt_sur.reorth = 1;

% Lan + Diag_Corr
opt2.cg_maxit = 1000; opt2.cg_tol = 1e-3; opt2.replace_diag = 1;
opt2.ldB2_lan = 1; opt2.ldB2_lan_reorth = 1; opt2.ldB2_lan_kmax = 100;

% Chebyshev
opt3.cg_maxit = 1000; opt3.cg_tol = 1e-3; opt3.replace_diag = 0; opt3.ldB2_cheby = 1;
inf3 = @(varargin)infGaussLik(varargin{:},opt3);

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
    hyp0 = struct('mean', [], 'cov', 0.9*hyp.cov,'lik', 1.2*hyp.lik);
    Y = generate_data(X,opt_Y);
    data(:,nrun) = Y;
    hyp_recover = cell(6,1);
    NLK = cell(6,1);
    
    for j = 1:1
        
        tic;
        sur = build_surrogate(covg,X,opt_sur);
        time(1) = time(1) + toc;
        % Sur + Lan + Diag_Corr
        opt1.cg_maxit = 1000; opt1.cg_tol = 1e-3; opt1.replace_diag = 1; 
        opt1.ldB2_sur = sur;
        inf1 = @(varargin)infGaussLik(varargin{:},opt1);
        % Sur + Lan + Diag_Corr
        tic;
        temp1 = minimize(hyp0,@gp,-30,inf1,means,covg,lik,X,Y);
        hyp_sur(j,:) = exp([temp1.cov',temp1.lik]);
        time(2) = time(2) + toc;
        [~,nlZ,dnlZ] = infGaussLik(temp1,means,cov,lik,X,Y,opt6);
        NLK{1}(j,:) = [nlZ,dnlZ.cov',dnlZ.lik];
    end
    hyp_recover(1) ={hyp_sur};
    
    
    for j = 1:1
        % Lan + Diag_Corr
        opt2.ldB2_lan_hutch = sign(randn(N,10));
        inf2 = @(varargin)infGaussLik(varargin{:},opt2);
        tic;
        temp2 = minimize(hyp0,@gp,-30,inf2,means,covg,lik,X,Y);
        hyp_lan(j,:) = exp([temp2.cov',temp2.lik]);
        time(3) = time(3) + toc;
        [~,nlZ,dnlZ] = infGaussLik(temp2,means,cov,lik,X,Y,opt6);
        NLK{2}(j,:) = [nlZ,dnlZ.cov',dnlZ.lik];
    end
    hyp_recover(2) ={hyp_lan};
    
    %{
    tic;
    temp3 = minimize(hyp0,@gp,-50,inf3,means,covg,lik,X,Y);
    hyp_recover(3) = {exp([temp3.cov',temp3.lik])};
    time(4) = time(4) + toc;
    [~,nlZ,dnlZ] = infGaussLik(temp3,means,cov,lik,X,Y,opt6);
    NLK(3) = {[nlZ,dnlZ.cov',dnlZ.lik]};
    %}
    
    tic;
    temp4 = minimize(hyp0,@gp,-30,inf4,means,covg,lik,X,Y);
    hyp_recover(4) = {exp([temp4.cov',temp4.lik])};
    time(5) = time(5) + toc;
    [~,nlZ,dnlZ] = infGaussLik(temp4,means,cov,lik,X,Y,opt6);
    NLK(4) = {[nlZ,dnlZ.cov',dnlZ.lik]};
    
    
    tic;
    temp5 = minimize(hyp0,@gp,-30,inf5,means,covf,lik,X,Y);
    hyp_recover(5) = {exp([temp5.cov',temp5.lik])};
    time(6) = time(6) + toc;
    [~,nlZ,dnlZ] = infGaussLik(temp5,means,cov,lik,X,Y,opt6);
    NLK(5) = {[nlZ,dnlZ.cov',dnlZ.lik]};
    
    
    tic;
    temp6 = minimize(hyp0,@gp,-30,inf6,means,cov,lik,X,Y);
    hyp_recover(6) = {exp([temp6.cov',temp6.lik])};
    time(7) = time(7) + toc;
    [~,nlZ,dnlZ] = infGaussLik(temp6,means,cov,lik,X,Y,opt6);
    NLK(6) = {[nlZ,dnlZ.cov',dnlZ.lik]};
    result{nrun} = hyp_recover;
    result_NLK{nrun} = NLK;
    
end


%{
result = exp([hyp.cov;hyp.lik]);
% result = [exp([hyp.cov';hyp.lik]);hyp.mean'];
%[result hyp_est' hyp_exact'] % Use this line when only one run
result = [result, mean(hyp_est)', std(hyp_est)', mean(hyp_grid)', std(hyp_grid)',...
    mean(hyp_sparse)', std(hyp_sparse)'];% mean(hyp_exact)', std(hyp_exact)'];
T = array2table(result,'VariableNames',{'True','Sur_Lan_Grid','Error1',...
    'Grid','Error2','Sparse','Error3'})%,'Exact','Error4'})
time(2:5) = time(2:5)/5;
method = {'Sur','Grid','Sparse','Exact'};
for k = 1:4
    fprintf('Time for %s optimization is %.3f.\n',method{k},time(k+1));
end
%}