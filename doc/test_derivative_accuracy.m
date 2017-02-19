% Cross-section plots for the derivative close to the global minimum
%
% We generate a GP with known hypers and look at the derivatives around
% these values by changing one of the hypers. This gives us an idea about
%
%

clc
clear all
close all

addpath('../')
startup

% Input data
N = 1000; % Size of dataset
Ngrid = 500; % Number of grid points
Nlin = 30; % Size of linspace
Nruns = 3; % Number of runs with different probe vectors
cheb_iter = 250; % Number of Chebyshev iterations
lan_iter = 150; % Number of Lanczos iterations

% kernel == 1 : RBF
% kernel == 2 : Matern
kernel = 1;
if kernel == 1; fprintf('Using RBF kernel\n'); else fprintf('Using Matern kernel\n'); end
if kernel == 1
    cov = {@covSEiso};
    der = {@derivative_RBF};
elseif kernel == 2
    cov = {@(varargin)covMaterniso(1,varargin{:})};
    der = {@derivative_OU};
end

% do_apx == 1: Random + APX
% do_axp == 0: Linspace
do_apx = 1;
if do_apx == 1; fprintf('Using Random + APX\n'); else fprintf('Using Linspace\n'); end
if do_apx == 1
    X = 4*rand(N, 1);
    xg = apxGrid('create', X, true, Ngrid);
    covg = {@apxGrid,cov,xg};
else
    X = linspace(0, 4, N)';
end

probe_vecs = {};
for i=1:Nruns
    probe_vecs{i} = sign(randn(N,ceil(log(N))));
end

% True hyper parameters
ell_true = 0.1; sf_true = 1; sigma_true = 0.1;
hyp_true = struct('mean', [], 'cov', log([ell_true, sf_true]), 'lik', log(sigma_true));
bounds = log([0.05,0.5,0.05; 0.2,2,0.2]);
ell_vec = exp(linspace(bounds(1,1), bounds(2,1), Nlin));
sf_vec = exp(linspace(bounds(1,2), bounds(2,2), Nlin));
sigma_vec = exp(linspace(bounds(1,3), bounds(2,3), Nlin));
mu = zeros(N, size(X, 2));

% Generate a GP the we want to recover
if do_apx
    opt.cg_maxit = 500; opt.cg_tol = 1e-5; opt.replace_diag = 1;
    K = apx(hyp_true, covg, X, opt);
    Kfull = K.mvm(eye(N));
    C = chol(0.5*(Kfull+Kfull')+exp(2*hyp_true.lik)*eye(N));
    f = C' * randn(size(C, 1), 1) + repmat(mu, 1, 1);
else
    D2 = pdist2(X, X).^2;
    K = cov{:}(log([ell_true; sf_true]), X) + sigma_true^2 * eye(N);
    K = 0.5*(K + K');
    C = cholcov(K);
    f = C' * randn(size(C, 1), 1) + repmat(mu, 1, 1);
end

% Look at the different cross-section by varying parameter i
vals = zeros(4, Nlin, 3, Nruns); % method, linspace, hyper
derivs = zeros(4, Nlin, 3, Nruns); % method, linspace, hyper
term1_vals = zeros(Nlin, 3); % linspace, hyper
term1_derivs = zeros(Nlin, 3); % linspace, hyper
term1_vals_apx = zeros(Nlin, 3); % linspace, hyper
term1_derivs_apx = zeros(Nlin, 3); % linspace, hyper

for i=1:3 % What hyper to vary [ell, sf, sigma]
    for j=1:Nlin % Loop over the linspace of the varying hyper
        fprintf('(%d,%d) ', i, j)
        
        ell = ell_true; sf = sf_true; sigma = sigma_true;
        if i==1 % Vary ell, compute derivatives in ell
            ell = ell_vec(j);
        elseif i==2 % Vary sf, compute derivatives in sf
            sf = sf_vec(j);
        elseif i==3 % Vary sigma, compute derivatives in sigma
            sigma = sigma_vec(j);
        end
        hyp = struct('cov', log([ell sf]), 'lik', log(sigma));
        sn2 = exp(2*hyp.lik); % Extract sigma^2 for convenience
        
        % Exact derivatives
        if do_apx
            opt.cg_maxit = 500; opt.cg_tol = 1e-5; opt.replace_diag = 1;
            K = apx(hyp, covg, X, opt);
            [~,solveKiW] = K.fun(ones(N,1)/exp(2*hyp.lik));
            Kfull = K.mvm(eye(N));
            C = chol(0.5*(Kfull+Kfull')/exp(2*hyp.lik)+eye(N));
            vals(1, j, i, 1) = sum(log(diag(C)));
        else
            K = cov{:}(hyp.cov, X);
            Ksigma = (K+K')/2/sigma^2+eye(N);
            c = Ksigma(:,1); % For Toeplitz
            C = cholcov(Ksigma);
            vals(1, j, i, 1) = sum(log(diag(C)));
        end
            alpha = solveKiW(f);
            term1_vals(j, i) = f'*alpha;
        
        if kernel == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%% RBF kernel
            if do_apx
                dKxg = derivative_RBF(hyp,xg{:},1);
                dKxg = dKxg(eye(Ngrid));
                if i == 1
                    dK = K.Mx * dKxg(:,1:Ngrid) * K.Mx';
                    diag_old = diag(dK);
                    dK = (dK - diag(diag(dK)))/sn2;
                    dKg = @(x) (K.Mx * toeplitzmult(dKxg(:,1),K.Mx'*x) - bsxfun(@times,diag_old,x))/sn2;
                elseif i == 2
                    dK = K.Mx * dKxg(:,Ngrid+1:end) * K.Mx';
                    diag_old = diag(dK);
                    dK = (dK - diag(diag(dK)) + 2 * sf^2 * eye(N))/sn2;
                    dKg = @(x) (K.Mx * toeplitzmult(dKxg(:,Ngrid+1),K.Mx'*x) + bsxfun(@times,2*sf^2-diag_old,x))/sn2;
                elseif i == 3
                    dK = - 2 * Kfull/sn2;
                    dKg = @(x) -2*K.mvm(x)/sn2;
                end
            else
                if i == 1 % ell
                    dK = (1/ell^2) * D2 .* K;
                elseif i == 2 % sf
                    dK = 2 * K;
                elseif i == 3 % sigma
                    dK = 2 * sigma^2 * eye(N);
                end
            end
        elseif kernel == 2 %%%%%%%%%%%%%%%%%%%%%%%% Matern kernel
            if do_apx
                dKxg = derivative_OU(hyp,xg{:},1);
                dKxg = dKxg(eye(Ngrid));
                if i == 1
                    dK = K.Mx * dKxg(:,1:Ngrid) * K.Mx';
                    diag_old = diag(dK);
                    dK = (dK - diag(diag(dK)))/sn2;
                    dKg = @(x) (K.Mx * toeplitzmult(dKxg(:,1),K.Mx'*x) - bsxfun(@times,diag_old,x))/sn2;
                elseif i == 2
                    dK = K.Mx * dKxg(:,Ngrid+1:end) * K.Mx';
                    diag_old = diag(dK);
                    dK = (dK - diag(diag_old) + 2 * sf^2 * eye(N))/sn2;
                    dKg = @(x) (K.Mx * toeplitzmult(dKxg(:,Ngrid+1),K.Mx'*x) + bsxfun(@times,2*sf^2-diag_old,x))/sn2;
                elseif i == 3
                    dK = - 2 * Kfull/sn2;
                    dKg = @(x) -2*K.mvm(x)/sn2;
                end
            else
                if i == 1 % ell
                    dK = (1/ell) * sqrt(D2) .* K;
                elseif i == 2 % sf
                    dK = 2 * K;
                elseif i == 3 % sigma
                    dK = 2 * sigma^2 * eye(N);
                end
            end
        end
            
        derivs(1, j, i, 1) = trace(solveKiW(dK));
        term1_derivs(j, i) = alpha'*dK*alpha;
        
        %{
        % Set up the grid
        dK = der{:}(hyp, X);
        if do_apx
            Kx = apx(hyp, {@apxGrid, {cov}, xg}, X);
            dKxg = der{:}(hyp, xg{:}, 1);
            dKg = @(x)[Kx.Mx*dKxg(Kx.Mx'*x), 2*sn2*x]; % For grid
            MAKE GRID USE A DIAGONAL CORRECTION FOR THE DERIVATIVES
            
            % Compute first term for APX and derivatives
            term1_vals_apx(j, i) = ????;
            term1_derivs_apx(j, i) = ????;
        end
        
        %%%% Make diagonal correction for APX
        if do_apx
            dd = sn2*ones(N, 1);
            for m=1:N
                em = sparse(N, 1); em(m) = 1;
                dd(m) = Ksigma(m,m) - em'*Kx.mvm(em);
            end
        end
        %}
        
        for run=1:Nruns
            fprintf('%d ', run)
            % Lanczos
            if do_apx
                %[vals(2,j,i,run), temp] = logdet_lanczos(@(x)Kx.mvm(x)+bsxfun(@times,dd,x), N, probe_vecs{run}, dKg, lan_iter, 1);
                [vals(2,j,i,run), temp] = logdet_lanczos(@(x)K.mvm(x)+sigma^2*x, N, probe_vecs{run}, dKg, lan_iter, 1);
                derivs(2,j,i,run) = temp;
            else
                [vals(2,j,i,run), temp] = logdet_lanczos(c, N, probe_vecs{run}, dK, lan_iter, 1);
                derivs(2,j,i,run) = temp(i);
            end
            
            
            % Cheb
            if do_apx
                %[vals(3,j,i,run), temp] = logdet_cheb(@(x)Kx.mvm(x)+bsxfun(@times,dd,x), N, probe_vecs{run}, cheb_iter, dKg, 3, sqrt(sigma^2));
                [vals(3,j,i,run), temp] = logdet_cheb(@(x)K.mvm(x)+sigma^2*x, N, probe_vecs{run}, cheb_iter, dKg, 1, sqrt(sigma^2));
                derivs(3,j,i,run) = temp;
            else
                [vals(3,j,i,run), temp] = logdet_cheb(@(x)toeplitzmult(c,x), N, probe_vecs{run}, cheb_iter, dK, 3, sqrt(sigma^2));
                derivs(3,j,i,run) = temp(i);
            end
            
        end
        fprintf('\n')
        
        % Scaled eig
        if do_apx
            [vals(4,j,i,1), temp] = logdet_gpml(hyp, {@apxGrid, {cov}, xg}, X, struct('replace_diag',1));
            derivs(4,j,i,1) = temp(i);
        end
    end
end

% Save the data
if kernel == 1 && do_apx == 0; save('deriv1d_rbf.mat');
elseif kernel == 1 && do_apx == 1; save('deriv1d_rbf_apx.mat');
elseif kernel == 2 && do_apx == 0; save('deriv1d_matern.mat');
elseif kernel == 2 && do_apx == 1; save('deriv1d_matern_apx.mat');
end

% Log marginal likelihood
figure
for i=1:3
    subplot(2,3,i)
    if i==1; xdata=ell_vec; elseif i==2; xdata=sf_vec; elseif i==3; xdata=sigma_vec; end
    
    % Lanczos, Cheb
    colors = {'r', 'g'};
    for j=1:2
        if do_apx
            data = bsxfun(@plus,-0.5*squeeze(vals(j+1,:,i,:)),-0.5*term1_vals_apx(:,i))-N/2*log(2*pi);
        else
            data = bsxfun(@plus,-0.5*squeeze(vals(j+1,:,i,:)),-0.5*term1_vals(:,i))-N/2*log(2*pi);
        end
        if size(data,1)==1; data=data'; end
        m = mean(data, 2);
        e = std(data, 0, 2);
        errorbar(xdata, m, e, colors{j}, 'LineWidth', 2)
        if j==1; set(gca,'xscale','log'); hold on; end
    end
    
    % Exact, Scaled eig
    data = bsxfun(@plus,-0.5*squeeze(vals(1,:,i,1))',-0.5*term1_vals(:,i))-N/2*log(2*pi);
    semilogx(xdata, data,'k.','MarkerSize', 30)
    if do_apx
        data = bsxfun(@plus,-0.5*squeeze(vals(4,:,i,1))',-0.5*term1_vals_apx(:,i))-N/2*log(2*pi);
        semilogx(xdata, data,'bx','MarkerSize', 20)
    end
    
    if i==1; title('loglik','fontsize',24); xlabel('ell','fontsize',24);
    elseif i==2; title('loglik','fontsize',24); xlabel('sf','fontsize',24);
    else title('loglik','fontsize',24); xlabel('sigma','fontsize',24);
    end
    if i==1; legend('Lanczos', 'Chebyshev', 'Exact', 'Scaled eig'); end
    axis tight
    set(gca, 'fontsize', 20)
    
    subplot(2,3,i+3)
    % Lanczos, Cheb
    colors = {'r', 'g'};
    for j=1:2
        if do_apx
            data = bsxfun(@plus,-0.5*squeeze(derivs(j+1,:,i,:)),0.5*term1_derivs_apx(:,i));
        else
            data = bsxfun(@plus,-0.5*squeeze(derivs(j+1,:,i,:)),0.5*term1_derivs(:,i));
        end
        if size(data,1)==1; data=data'; end
        m = mean(data, 2);
        e = std(data, 0, 2);
        errorbar(xdata, m, e, colors{j}, 'LineWidth', 2)
        if j==1; set(gca,'xscale','log'); hold on;  end
    end
    
    % Exact, Scaled eig
    data = bsxfun(@plus,-0.5*squeeze(derivs(1,:,i,1))',0.5*term1_derivs(:,i));
    semilogx(xdata, data,'k.','MarkerSize', 30)
    if do_apx
        data = bsxfun(@plus,-0.5*squeeze(derivs(4,:,i,1))',0.5*term1_derivs_apx(:,i));
        semilogx(xdata, data,'bx','MarkerSize', 20)
    end
    
    if i==1; title('dloglik/dell','fontsize',24); xlabel('ell','fontsize',24);
    elseif i==2; title('dloglik/dsf','fontsize',24); xlabel('sf','fontsize',24);
    else title('dloglik/dsigma','fontsize',24); xlabel('sigma','fontsize',24);
    end
    axis tight
    set(gca, 'fontsize', 20)
end


% Log-det
figure
for i=1:3
    subplot(2,3,i)
    if i==1; xdata=ell_vec; elseif i==2; xdata=sf_vec; elseif i==3; xdata=sigma_vec; end
    
    % Lanczos, Cheb
    colors = {'r', 'g'};
    for j=1:2
        data = squeeze(vals(j+1,:,i,:));
        if size(data,1)==1; data=data'; end
        m = mean(data, 2);
        e = std(data, 0, 2);
        errorbar(xdata, m, e, colors{j}, 'LineWidth', 2)
        if j==1; set(gca,'xscale','log'); hold on;  end
    end
    
    % Exact, Scaled eig
    data = squeeze(vals(1,:,i,1));
    semilogx(xdata, data,'k.','MarkerSize', 30)
    if do_apx
        data = squeeze(vals(4,:,i,1));
        semilogx(xdata, data,'bx','MarkerSize', 20)
    end
    
    if i==1; title('logdet','fontsize',24); xlabel('ell','fontsize',24);
    elseif i==2; title('logdet','fontsize',24); xlabel('sf','fontsize',24);
    else title('logdet','fontsize',24); xlabel('sigma','fontsize',24);
    end
    if i==1; legend('Lanczos', 'Chebyshev', 'Exact', 'Scaled eig'); end
    axis tight
    set(gca, 'fontsize', 20)
    
    subplot(2,3,i+3)
    % Lanczos, Cheb
    colors = {'r', 'g'};
    for j=1:2
        data = squeeze(derivs(j+1,:,i,:));
        if size(data,1)==1; data=data'; end
        m = mean(data, 2);
        e = std(data, 0, 2);
        errorbar(xdata, m, e, colors{j}, 'LineWidth', 2)
        if j==1; set(gca,'xscale','log'); hold on;  end
    end
    
    % Exact, Scaled eig
    data = squeeze(derivs(1,:,i,1));
    semilogx(xdata, data,'k.','MarkerSize', 30)
    if do_apx
        data = squeeze(derivs(4,:,i,1));
        semilogx(xdata, data,'bx','MarkerSize', 20)
    end
    
    if i==1; title('dlogdet/dell','fontsize',24); xlabel('ell','fontsize',24);
    elseif i==2; title('dlogdet/dsf','fontsize',24); xlabel('sf','fontsize',24);
    else title('dlogdet/dsigma','fontsize',24); xlabel('sigma','fontsize',24);
    end
    axis tight
    set(gca, 'fontsize', 20)
end
%}