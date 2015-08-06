function [D, D2] = dictionary_2layer(X, params, params2)


    % Extract patches
    patches = extract_patches(X, params);

    % Train 1st layer dictionary
    D = dictionary(patches, params);

    % Extract features
    chunksize = 100;
    m = size(X, 1);
    numchunks = ceil(m ./ 100);
    XL = cell(m, 1);
    for i = 1:numchunks
        batch = X((i-1) * chunksize + 1 : min([i * chunksize end]));
        [foo, L] = extract_features(batch, D, params);
        XL((i-1) * chunksize + 1 : min([i * chunksize end])) = L;
    end
        
    % Extract 2nd layer patches
    patches = extract_patches(XL, params2);

    % Train 2nd layer dictionary
    D2 = dictionary(patches, params2);
    
    % Loop over data chunks and extract features
    %patch_params = params2;
    %chunksize = 100;
    %m = size(X, 1);
    %numchunks = ceil(m ./ 100);
    %ppp = ceil(params2.npatches ./ numchunks);
    %patch_params.npatches = ppp;
    %patches2 = zeros(numchunks * ppp, params2.rfSize(1) * params2.rfSize(2) * params2.rfSize(3));
    
    %for i = 1:numchunks
        
    %    batch = X((i-1) * chunksize + 1 : min([i * chunksize end]));

        % Extract 1st layer features
    %    [f1, L1] = extract_features(batch, D, params);

        % Get patches    
    %    patches2((i-1) * ppp + 1 : i * ppp, :) = extract_patches(L1, patch_params);
    %end

    % Train 2nd layer dictionary
    %D2 = dictionary(patches2, params2);




