function [XC, XC2] = extract_features_2layer(X, D, D2, params, params2)

    
    % Wrapper for 2 layer feature extraction
    XC2 = zeros(size(X, 1), 5 * params2.nfeats);
    XC = zeros(size(X, 1), 14 * params.nfeats);
    
    % Break into chunks
    chunksize = 100;
    m = size(X, 1);
    numchunks = ceil(m ./ 100);
    
    for i = 1:numchunks
        
        batch = X((i-1) * chunksize + 1 : min([i * chunksize end]));
     
        % Get 1st layer features
        [f1, L1] = extract_features(batch, D, params);

        % Get 2nd layer features
        f2 = extract_features(L1, D2, params2);

        % Append
        XC2((i-1) * chunksize + 1 : min([i * chunksize end]), :) = f2;
        XC((i-1) * chunksize + 1 : min([i * chunksize end]), :) = f1;

    end
    disp(' ');




