function D = whiten2(patches, params)

    C = cov(patches);
    D.mean = mean(patches);
    [V,E] = eig(C);
    D.whiten = V * diag(sqrt(1./(diag(E) + params.gamma))) * V';
