function X = best_slhd(n, d, ntrials, extra)
    %BEST_SLHD Generates several symmetric Latin hypercubes and picks the
    %   one that maximizes the minimum pairwise distance and minimizes
    %   the maximum correlation.
    %
    % David Eriksson, dme65@cornell.edu
    %

    % (1) Generate all the hypercubes and compute the correlation and
    % minimum distance

    X = zeros(n, d, ntrials);
    min_dists = zeros(ntrials, 1);
    corrs = zeros(ntrials, 1);
    for i=1:ntrials
        if nargin == 3
            XX = SLHD(d, n);
        else
            XX = SLHD(d, n - size(extra,1));
            XX = [XX; extra];
        end
        dist = pdist2(XX, XX);
        dist(1:n+1:end) = inf;
        min_dists(i) = min(dist(:));
        cvals = abs(corr(XX, XX));
        cvals(1:d+1:end) = 0;
        corrs(i) = sqrt(sum(cvals(:).^2));
        X(:,:,i) = XX;
    end

    % (2) Rescale the values to [0, 1]

    unit_min_dists = (min_dists - min(min_dists))/(max(min_dists) - min(min_dists));
    unit_corrs = 1 - (corrs - min(corrs))/(max(corrs) - min(corrs));

    % (3) Pick the best one

    weight = 0.5; % Really a mult-objective optimization problem?
    [~, best_ind] = max(weight*unit_min_dists + (1-weight)*unit_corrs);
    X = X(:, :, best_ind);
end
