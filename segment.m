function yhat = segment(im, mask, model, D, D2, params, params2, scaleparams)

    % Pre-process image
    up = [size(im, 1) size(im, 2)];
    im = pyramid(im, params);
    if max(mask(:)) > 1; mask = mask ./ 255; end

    % Extract first module feature maps
    L = extract_features(im, D, params);

    % Pre-process feature maps
    Lp = preproc2(L, params2);

    % Extract second module feature maps
    L2 = extract_features(Lp, D2, params2);
    clear Lp

    % Upsample
    L = upsample(L, params.numscales, up);
    L2 = upsample(L2, params.numscales, up);

    % Label each pixel
    yhat = annotate(cat(3, L{1}, L2{1}), model, mask, scaleparams);
    %yhat = annotate(L{1}, model, mask, scaleparams);
    
