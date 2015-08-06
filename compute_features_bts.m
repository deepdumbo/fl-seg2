function F = compute_features_bts(volumes, class, grade, D, D2, params, params2)

    % Loop over all volumes and compute
    F = cell(length(volumes), 1);
    sample = 1;
    for i = 1:length(volumes)

        % Load a volume
        I = load_braintumor(class, grade, volumes(i));
        [x,y,z] = size(I);
        up = [x y];
        I = I(:,:,sample:sample:end,:);
        I = pyramid_4d(I./255, params);

        % Compute 1st module features
        L = extract_features(I, D, params);

        % Pre-process (padding)
        Lp = preproc2(L, params);

        % Compute 2nd module features
        L2 = extract_features(Lp, D2, params2);

        % Upsample
        L = upsample(L, params.numscales, up);
        L2 = upsample(L2, params2.numscales, up);

        % Append to F
        Z = zeros(x, y, length(L), params.numscales * (params.nfeats + params2.nfeats));
        for j = 1:length(L)
            Z(:,:,j,:) = cat(3, L{j}, L2{j});
        end
        F{i} = Z;
        clear Z
        disp(' ');

    end

