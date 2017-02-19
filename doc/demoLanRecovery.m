clear, close all, write_fig = 0; 
N = 1000; ngrid = 500;

% Choose data points
X = sort(4*rand(N,1)-2,'ascend');
%X = linspace(0,4,N)';

% Generate data
hyp = struct('mean', [], 'cov', log([0.2;0.8]), 'lik', log(0.1));
opt_Y.hyp = hyp; opt_Y.type = 'RBF';
Y = generate_data(X,opt_Y);

% Setup SKI and FITC
xg = apxGrid('create',X,true,ngrid);
xu = linspace(-2,2,ngrid)';
%cov = {@(varargin)covMaterniso(1,varargin{:})};
cov = {@covSEiso};
covg = {@apxGrid,cov,xg};
covf = {@apxSparse,cov,xu};
lik = {@likGauss};
means = {@meanZero};

time = zeros(1,7);

% Build surrogate 
opt_sur.npts = 200; opt_sur.ntrials = 1000; opt_sur.param = {{'cov','lik'},[2,1]};
opt_sur.bounds = log([5e-2,0.2,1e-2;2,2,0.5]); opt_sur.method = 'lanczos';
opt_sur.cg_maxit = 500; opt_sur.cg_tol = 1e-5; opt_sur.replace_diag = 1; opt_sur.nZ = 30;

tic;
sur = build_surrogate(covg,X,opt_sur);
time(1) = toc;

% Sur + Lan + Diag_Corr
opt1.cg_maxit = 500; opt1.cg_tol = 1e-3; opt1.replace_diag = 1; %opt1.ldB2_lan_hutch = sign(randn(N,10));
opt1.ldB2_sur = sur;  %opt1.ldB2_lan_reorth = 0; %opt1.ldB2_lan_kmax = 100;
inf1 = @(varargin)infGaussLik(varargin{:},opt1);

% Lan + Diag_Corr
opt2.cg_maxit = 500; opt2.cg_tol = 1e-3; opt2.replace_diag = 1;
opt2.ldB2_lan = 1; opt2.ldB2_lan_hutch = sign(randn(N,30));
opt2.ldB2_lan_reorth = 1; opt2.ldB2_lan_kmax = 100;
inf2 = @(varargin)infGaussLik(varargin{:},opt2);

% Apx + Diag_Corr
opt3.cg_maxit = 500; opt3.cg_tol = 1e-3; opt3.replace_diag = 1;
inf3 = @(varargin)infGaussLik(varargin{:},opt3);

% Apx + No_Diag_Corr
opt4.cg_maxit = 500; opt4.cg_tol = 1e-3; opt4.replace_diag = 0;
inf4 = @(varargin)infGaussLik(varargin{:},opt4);

% FITC
opt5.cg_maxit = 500; opt5.cg_tol = 1e-3;
inf5 = @(varargin)infGaussLik(varargin{:},opt5);

% Exact
opt6.cg_maxit = 500; opt6.cg_tol = 1e-3; 
inf6 = @(varargin)infGaussLik(varargin{:},opt6);

for nrun = 1:1
    hyp0 = struct('mean', [], 'cov', 0.9*hyp.cov,'lik', 0.9*hyp.lik);
    
    % Sur + Lan + Diag_Corr
    tic;
    temp1 = minimize(hyp0,@gp,-100,inf1,means,covg,lik,X,Y);
    hyp_sur(nrun,:) = exp([temp1.cov',temp1.lik]);
    time(2) = time(2) + toc;
    
    % Lan + Diag_Corr
    tic;
    temp2 = minimize(hyp0,@gp,-100,inf2,means,covg,lik,X,Y);
    hyp_lan(nrun,:) = exp([temp2.cov',temp2.lik]);
    time(3) = time(3) + toc;
   
    tic;
    temp3 = minimize(hyp0,@gp,-100,inf3,means,covg,lik,X,Y);
    hyp_gdc(nrun,:) = exp([temp3.cov',temp3.lik]);
    time(4) = time(4) + toc;
    
    tic;
    temp4 = minimize(hyp0,@gp,-100,inf4,means,covg,lik,X,Y);
    hyp_grid(nrun,:) = exp([temp4.cov',temp4.lik]);
    time(5) = time(4) + toc;
    
    tic;
    temp5 = minimize(hyp0,@gp,-100,inf5,means,covf,lik,X,Y);
    hyp_gdc(nrun,:) = exp([temp5.cov',temp5.lik]);
    time(6) = time(6) + toc;
    
    tic;
    temp6 = minimize(hyp0,@gp,-100,inf6,means,cov,lik,X,Y);
    hyp_exact(nrun,:) = exp([temp6.cov',temp6.lik]);
    time(7) = time(7) + toc;
end


[post,nlZ,dnlZ] = infGaussLik(temp1,means,cov,lik,X,Y,opt6);[nlZ,dnlZ.cov',dnlZ.lik]

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
%}