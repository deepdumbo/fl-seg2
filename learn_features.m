function [D, D2, X, labels] = learn_features(params, params2)

    annotations = '/cshome/dana/foisy4/DATA/VESSEL12/Annotations/VESSEL12_%d_Annotations.csv';

    % Load volumes, annotations and pre-process
    disp('Loading and pre-processing data...')
    V = cell(3, 1);
    A = cell(3, 1);
    Vlist = cell(3, 1);
    for i = 1:3
        I = load_vessels(20+i);
        %A{i} = load(sprintf('/home/haltair/Research/vision/images/Vessel/Annotations/VESSEL12_%d_Annotations.csv', 20+i));
        A{i} = load(sprintf(annotations, 20+i));
        V{i} = pyramid(I, params);
        Vlist{i} = imagelist(A{i}, params.numscales);
        clear I;
    end
    
    % Extract first module patches
    patches = extract_patches([V{1}; V{2}; V{3}], params);

    % Train first module dictionary
    D = dictionary(patches, params);

    % Compute first module feature maps on slices with annotations
    disp('Extracting first module feature maps...')
    L = cell(3, 1);
    Lp = cell(3, 1);
    Up = cell(3, 1);
    for i = 1:3
        L{i} = extract_features(V{i}(Vlist{i}), D, params);
        Lp{i} = preproc2(L{i}, params2);
        r = randi(length(L{i}), 100, 1);
        Up{i} = extract_features(V{i}(r), D, params);
        Up{i} = preproc2(Up{i}, params2);
    end
    

    % Extract second module patches
    patches = extract_patches([Lp{1}; Lp{2}; Lp{3}; Up{1}; Up{2}; Up{3}], params2);
    clear Up

    % Train second module dictionary
    D2 = dictionary(patches, params2);

    % Compute second module feature maps
    disp('Extracting second module feature maps...')
    L2 = cell(3, 1);
    for i = 1:3
        L2{i} = extract_features(Lp{i}, D2, params2);
    end
    clear Lp

    % Upsample all feature maps
    disp('Upsampling feature maps...')
    for i = 1:3
        L{i} = upsample(L{i}, params.numscales, params.upsample);
        L2{i} = upsample(L2{i}, params.numscales, params.upsample);
    end

    % Compute pixel features for classification
    disp('Computing pixel-level features...')
    X = []; labels = [];
    for i = 1:3
        [tr, tl] = convert(L{i}, A{i}, Vlist{i}(params.numscales:params.numscales:end)/params.numscales);
        tr2 = convert(L2{i}, A{i}, Vlist{i}(params.numscales:params.numscales:end)/params.numscales);
        X = [X; [tr tr2]];
        labels = [labels; tl];
    end
        
