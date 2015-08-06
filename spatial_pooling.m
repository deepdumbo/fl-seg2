function features = spatial_pooling(XC, nrows, ncols, nmaps, gridsize)


    numtiles = gridsize(1) * gridsize(2);
    features = zeros(size(XC, 1), numtiles * nmaps);
    Q = zeros(numtiles, nmaps);
    r = round(nrows/gridsize(1));
    c = round(ncols/gridsize(2));
    p = 1;
    
    % Check for problems
    if r * gridsize(1) - 1 > nrows
        r = floor(nrows/gridsize(1));
    elseif c * gridsize(2) - 1 > ncols
        c = floor(ncols/gridsize(2));
    end
    
    for i = 1:size(XC, 1) 
        patches = reshape(XC(i,:), [nrows ncols nmaps]);
        index = 1;
        for j = 1:gridsize(2)-1
            for k = 1:gridsize(1)-1
                region = patches((k-1) * r + 1: k * r, (j-1) * c + 1: j * c, :);
                Q(index, :) = (sum(sum(region.^p,1),2)).^(1/p);
                index = index + 1;
            end
        end
        for j = 1: gridsize(2) - 1
            region = patches((gridsize(1)-1) * r + 1: end, (j-1) * c + 1: j * c, :);
            Q(index, :) = (sum(sum(region.^p,1),2)).^(1/p);
            index = index + 1;
        end
        for k = 1: gridsize(1) - 1
            region = patches((k-1) * r + 1: k * r, (gridsize(2)-1) * c + 1: end, :);
            Q(index, :) = (sum(sum(region.^p,1),2)).^(1/p);
            index = index + 1;
        end
        region = patches((gridsize(1)-1) * r + 1: end, (gridsize(2)-1) * c + 1: end, :);
        Q(index, :) = (sum(sum(region.^p,1),2)).^(1/p);
        features(i,:) = Q(:)';
    end


end

