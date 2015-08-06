function [patches, params] = extract_patches(V, params)

    
    % Parameters
    rfSize = params.rfSize;
    npatches = params.npatches;

    % Main loop
    %patches = zeros(npatches, rfSize(1) * rfSize(2) * size(squeeze(V{1}),3));
    patches = zeros(npatches, rfSize(1) * rfSize(2) * rfSize(3));
    disp('Extracting patches...');
    for i=1:npatches
        
        patch = double(V{mod(i-1,length(V))+1});
        patch = squeeze(patch);
        %patch = reshape(patch, [1 size(patch, 1) size(patch, 2)]);
        [nrows, ncols, nmaps] = size(patch);
       
        if (mod(i,10000) == 0) fprintf('Extracting patch: %d / %d\n', i, npatches); end

        % Extract random block
        r = random('unid', nrows - rfSize(1) + 1);
        c = random('unid', ncols - rfSize(2) + 1);
        patch = patch(r:r+rfSize(1)-1,c:c+rfSize(2)-1,:);
        patches(i,:) = patch(:)';
        
    end

    % Brightness and contrast normalization
    disp('Contrast normalization...');
    %patches = bsxfun(@minus, patches, mean(patches));
    %params.patchmean = mean(patches);
    %params.patchvar = sqrt(var(patches, [], 1) + 0.01); 
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2) + 10));

end

