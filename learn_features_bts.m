function [D, D2] = learn_features_bts(volumes, class, grade, params, params2)

    % Load and sample from volumes
    sample = 2;
    V = cell(length(volumes), 1);
    for i = 1:length(volumes)
        I = load_braintumor(class, grade, volumes(i));
        I = I(:,:,sample:sample:end,:);
        V{i} = pyramid_4d(I./255, params);
    end

    V

    flatten_cell(V)

    % Extract first module patches
    patches = extract_patches(flatten_cell(V)', params);
    
    % Train first module dictionary
    D = dictionary(patches, params);

    % Extract first module features
    L = cell(length(volumes), 1);
    for i = 1:length(volumes)
        L{i} = preproc2(extract_features(V{i}, D, params), params2);
    end

    % Extract second module patches
    patches = extract_patches(flatten_cell(L)', params2);

    % Train second module dictionary
    D2 = dictionary(patches, params2);    




        
