function [X, labels] = sample_bts(volumes, class, part, grade, F, numpts, params, params2)

    % Loop over each volume and sample
    sample = 1;
    X = zeros(2 * numpts * length(volumes), params.numscales * (params.nfeats + params2.nfeats));
    labels = zeros(2 * numpts * length(volumes), 1);
    for i = 1:length(volumes)
        
        % Load a volume
        [I, I_seg] = load_braintumor(class, grade, volumes(i));
        I = I(:,:,sample:sample:end,:);
        I_seg = I_seg(:,:,sample:sample:end,:);

        % Sample pixel features
        [x, lab] = sample_pixels(F{i}, part, I, I_seg, numpts, params, params2);

        % Append
        X((i-1) * 2 * numpts + 1 : i * 2 * numpts, :) = x;
        labels((i-1) * 2 * numpts + 1 : i * 2 * numpts) = lab;

    end

