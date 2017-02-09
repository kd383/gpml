% [Q, T] = moments_lanpro(A, n, Z, kmax)
%
% Grouped lanczos iterations with partial reorthogonalization for
% multiple vectors at once
%
% Input:
%   A: Matrix-vector multiplication operator
%   n: Dimension of A
%   Z: Starting vector as columns
%   kmax: Maximum Lanczos iteration
%
% Output:
%   Q: 3d-matrix of returned basis for Krylov subpsace of each vector
%   T: 3d-matrix of tridiagonal by each vector
%
function [Q,T] = moments_lanpro(A, n, Z, kmax)
    if nargin<4; kmax = max(10,n/10); end
    nZ = size(Z,2); % number of probe vectors

    % Set parameters
    delta = sqrt(eps/kmax); % Desired level of orthogonality.
    eta = eps^(3/4)/sqrt(kmax); % Level of orth. after reorthogonalization.
    fro = zeros(1,nZ); % full reorthogonalization

    % Rule-of-thumb estimate on the size of round-off terms:
    eps1 = sqrt(n)*eps/2; % Notice that {\bf u} == eps/2.
    gamma = 1/sqrt(2);
    anorm = cell(nZ,1);

    alpha = zeros(kmax+1,nZ);
    beta = zeros(kmax+1,nZ);
    Q = zeros(n,nZ,kmax);
    q = zeros(n,nZ);
    r = Z;
    beta(1,:) = sqrt(sum(r.^2));
    omega = zeros(kmax,nZ); omega_max = omega;  omega_old = omega;
    force_reorth = zeros(1,nZ);
    j0 = 1;

    for j = j0:kmax
        q_old = q;
        for l = 1:nZ
            if beta(j,l) == 0
                q(:,l) = r(:,l);
            else
                q(:,l) = r(:,l)/beta(j,l);
            end
        end
        Q(:,:,j) = q;
        u = A(q);
        r = u - bsxfun(@times,q_old,beta(j,:));
        alpha(j,:) = sum(q.*r);
        r = r - bsxfun(@times,q,alpha(j,:));


        % Extended local reorthogonalization:
        beta(j+1,:) = sqrt(sum(r.^2)); % Quick and dirty estimate.
        for l = 1:nZ
            if beta(j+1,l) < gamma*beta(j,l)
                if  j == 1
                    t1 = 0;
                    for i = 1:2
                        t = q(:,l)'*r(:,l);
                        r(:,l) = r(:,l)-q(:,l)*t;
                        t1 = t1+t;
                    end
                    alpha(j,l) = alpha(j,l) + t1;
                elseif j > 1
                    t1 = q_old(:,l)'*r(:,l);
                    t2 = q(:,l)'*r(:,l);
                    r(:,l) = r(:,l)  - (q_old(:,l)*t1 + q(:,l)*t2); % Add small terms together first to
                    if beta(j,l) ~= 0               % reduce risk of cancellation.
                        beta(j,l) = beta(j,l) + t1;
                    end
                    alpha(j,l) = alpha(j,l) + t2;
                end
                beta(j+1,l) = norm(r(:,l)); % Quick and dirty estimate.
            end

            if  beta(j+1,l)~=0
                anorm{l} = update_gbound(anorm{l},alpha(:,l),beta(:,l),j);
            end

            % Update omega-recurrence
            if j > 1 && ~fro(l) && beta(j+1,l) ~= 0
                [omega(:,l),omega_old(:,l)] = update_omega(omega(:,l),omega_old(:,l),j,alpha(:,l),beta(:,l),...
                                            eps1,anorm{l});
                omega_max(j,l) = max(abs(omega(:,l)));
            end

            % Reorthogonalize if required
            if j > 1 && (fro(l)  || force_reorth(l) || omega_max(j,l)>delta) && beta(j+1,l) ~= 0
                if fro(l)
                    int = 1:j;
                else
                    if force_reorth(l) == 0
                        force_reorth(l) = 1; % Do forced reorth to avoid spill-over from q_{j-1}.
                        int = compute_int(omega(:,l),j,delta,eta,0,0,0);
                    else
                        force_reorth(l) = 0;
                    end
                end
                [r(:,l),beta(j+1,l),junk] = reorth(Q(:,l,:),r(:,l),beta(j+1,l),int,gamma,0);
                omega(int,l) = eps1;
            else
                beta(j+1,l) = norm(r(:,l)); % compute norm accurately.
            end

            if  j < kmax && beta(j+1,l) < n*anorm{l}*eps  ,
            % If beta is "small" we deflate by setting the off-diagonals of T_k
            % to 0 and attempt to restart with a basis for a new
            % invariant subspace by replacing r with a random starting vector:
                beta(j+1,l) = 0;
                bailout = 1;
                for attempt=1:3
                    r(:,l) = rand(n,1)-0.5;
                    r(:,l) = A(r(:,l));
                    nrm=norm(r(:,l)); % not necessary to compute the norm accurately here.
                    int = 1:j;
                    [r(:,l),nrmnew,junk] = reorth(Q(:,l,:),r(:,l),nrm,int,gamma,0);
                    omega(int,l) = eps1;
                    if nrmnew > 0
                        bailout=0;
                        break;
                    end
                end
                if bailout
                    break;
                else
                    r(:,l)=r(:,l)/nrmnew; % Continue with new normalized r as starting vector.
                    force_reorth(l) = 1;
                    if delta > 0
                        fro(l) = 0;    % Turn off full reorthogonalization.
                    end
                end
            elseif j<kmax && ~fro(l) && beta(j+1,l)*delta < anorm{l}*eps1,
                fro(l) = 1;
            end
        end
    end

    if nZ == 1
        Q = squeeze(Q);
        T = diag(alpha(1:j))+diag(beta(2:j),-1)+diag(beta(2:j),1);
        %{
        if beta(j) > 1e-6
            warning('The Lanczos has not converged!');
        end
        %}
    else
        T = zeros(j,j,nZ);
        % Set up tridiagonal T_k in sparse matrix data structure.
        for l = 1:nZ
            T(:,:,l) = diag(alpha(1:j,l))+diag(beta(2:j,l),-1)+diag(beta(2:j,l),1);
            %{
                if beta(j,l) > 1e-6
                    warning('The Lanczos has not converged! Consider run more iterations.');
                end
            %}
        end
    end


function [omega,omega_old] = update_omega(omega, omega_old, j, ...
    alpha,beta,eps1,anorm)
% UPDATE_OMEGA:  Update Simon's omega_recurrence for the Lanczos vectors.
%
% [omega,omega_old] = update_omega(omega, omega_old,j,eps1,alpha,beta,anorm)
%

% Rasmus Munk Larsen, DAIMI, 1998.

% Estimate of contribution to roundoff errors from A*v
%   fl(A*v) = A*v + f,
% where ||f|| \approx eps1*||A||.
% For a full matrix A, a rule-of-thumb estimate is eps1 = sqrt(n)*eps.
T = eps1*anorm;
binv = 1/beta(j+1);

omega_old = omega;
% Update omega(1) using omega(0)==0.
omega_old(1)= beta(2)*omega(2)+ (alpha(1)-alpha(j))*omega(1) -  ...
    beta(j)*omega_old(1);
omega_old(1) = binv*(omega_old(1) + sign(omega_old(1))*T);
% Update remaining components.
k=2:j-2;
omega_old(k) = beta(k+1).*omega(k+1) + (alpha(k)-alpha(j)).*omega(k) ...
     + beta(k).*omega(k-1) - beta(j)*omega_old(k);
omega_old(k) = binv*(omega_old(k) + sign(omega_old(k))*T);
omega_old(j-1) = binv*T;
% Swap omega and omega_old.
temp = omega;
omega = omega_old;
omega_old = omega;
omega(j) =  eps1;


function anorm = update_gbound(anorm,alpha,beta,j)
%UPDATE_GBOUND   Update Gerscgorin estimate of 2-norm
%  ANORM = UPDATE_GBOUND(ANORM,ALPHA,BETA,J) updates the Gerscgorin bound
%  for the tridiagonal in the Lanczos process after the J'th step.
%  Applies Gerscgorins circles to T_K'*T_k instead of T_k itself
%  since this gives a tighter bound.

if j==1 % Apply Gerscgorin circles to T_k'*T_k to estimate || A ||_2
  i=j;
  % scale to avoid overflow
  scale = max(abs(alpha(i)),abs(beta(i+1)));
  alpha(i) = alpha(i)/scale;
  beta(i+1) = beta(i+1)/scale;
  anorm = 1.01*scale*sqrt(alpha(i)^2+beta(i+1)^2 + abs(alpha(i)*beta(i+1)));
elseif j==2
  i=1;
  % scale to avoid overflow
  scale = max(max(abs(alpha(1:2)),max(abs(beta(2:3)))));
  alpha(1:2) = alpha(1:2)/scale;
  beta(2:3) = beta(2:3)/scale;

  anorm = max(anorm, scale*sqrt(alpha(i)^2+beta(i+1)^2 + ...
      abs(alpha(i)*beta(i+1) + alpha(i+1)*beta(i+1)) + ...
      abs(beta(i+1)*beta(i+2))));
  i=2;
  anorm = max(anorm,scale*sqrt(abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
      beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
      abs(alpha(i)*beta(i+1))) );
elseif j==3
  % scale to avoid overflow
  scale = max(max(abs(alpha(1:3)),max(abs(beta(2:4)))));
  alpha(1:3) = alpha(1:3)/scale;
  beta(2:4) = beta(2:4)/scale;
  i=2;
  anorm = max(anorm,scale*sqrt(abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
      beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
      abs(alpha(i)*beta(i+1) + alpha(i+1)*beta(i+1)) + ...
      abs(beta(i+1)*beta(i+2))) );
  i=3;
  anorm = max(anorm,scale*sqrt(abs(beta(i)*beta(i-1)) + ...
      abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
      beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
      abs(alpha(i)*beta(i+1))) );
else
  % scale to avoid overflow
  %  scale = max(max(abs(alpha(j-2:j)),max(abs(beta(j-2:j+1)))));
  %  alpha(j-2:j) = alpha(j-2:j)/scale;
  %  beta(j-2:j+1) = beta(j-2:j+1)/scale;

  % Avoid scaling, which is slow. At j>3 the estimate is usually quite good
  % so just make sure that anorm is not made infinite by overflow.
  i = j-1;
  anorm1 = sqrt(abs(beta(i)*beta(i-1)) + ...
      abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
      beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
      abs(alpha(i)*beta(i+1) + alpha(i+1)*beta(i+1)) + ...
      abs(beta(i+1)*beta(i+2)));
  if isfinite(anorm1)
    anorm = max(anorm,anorm1);
  end
  i = j;
  anorm1 = sqrt(abs(beta(i)*beta(i-1)) + ...
      abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
      beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
      abs(alpha(i)*beta(i+1)));
  if isfinite(anorm1)
    anorm = max(anorm,anorm1);
  end
end


function int = compute_int(mu,j,delta,eta,LL,strategy,extra)
%COMPUTE_INT:  Determine which Lanczos vectors to reorthogonalize against.
%
%      int = compute_int(mu,eta,LL,strategy,extra))
%
%   Strategy 0: Orthogonalize vectors v_{i-r-extra},...,v_{i},...v_{i+s+extra}
%               with nu>eta, where v_{i} are the vectors with  mu>delta.
%   Strategy 1: Orthogonalize all vectors v_{r-extra},...,v_{s+extra} where
%               v_{r} is the first and v_{s} the last Lanczos vector with
%               mu > eta.
%   Strategy 2: Orthogonalize all vectors with mu > eta.
%
%   Notice: The first LL vectors are excluded since the new Lanczos
%   vector is already orthogonalized against them in the main iteration.

%  Rasmus Munk Larsen, DAIMI, 1998.

if (delta<eta)
  error('DELTA should satisfy DELTA >= ETA.')
end
switch strategy
  case 0
    I0 = find(abs(mu(1:j))>=delta);
    if length(I0)==0
      [mm,I0] = max(abs(mu(1:j)));
    end
    int = zeros(j,1);
    for i = 1:length(I0)
      for r=I0(i):-1:1
	if abs(mu(r))<eta | int(r)==1
	  break;
	else
	  int(r) = 1;
	end
      end
      int(max(1,r-extra+1):r) = 1;
      for s=I0(i)+1:j
	if abs(mu(s))<eta | int(s)==1
	  break;
	else
	  int(s) = 1;
	end
      end
      int(s:min(j,s+extra-1)) = 1;
    end
    if LL>0
      int(1:LL) = 0;
    end
    int = find(int);
  case 1
    int=find(abs(mu(1:j))>eta);
    int = max(LL+1,min(int)-extra):min(max(int)+extra,j);
  case 2
    int=find(abs(mu(1:j))>=eta);
end
int = int(:);