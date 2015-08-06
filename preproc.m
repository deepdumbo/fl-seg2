function [images, M, V] = preproc(I, params)

    % Pad each slice, so valid convolutions will preserve size
    images = padarray(I, [(params.rfSize(1) - 1) / 2 (params.rfSize(2) - 1) / 2], 'replicate');
    
    % Contrast normalize
    %Im = reshape(images, size(images, 1), size(images, 2) * size(images, 3));
    %Im = bsxfun(@rdivide, bsxfun(@minus, Im, mean(Im,2)), sqrt(var(Im,[],2) + 10));
    %images = reshape(Im, size(images, 1), size(images, 2), size(images, 3));
        
    % Convert into cell
    images = squeeze(mat2cell(images, size(images, 1), size(images, 2), ones(1,size(images,3))));

end

