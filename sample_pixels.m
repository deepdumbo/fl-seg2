function [X, labels] = sample_pixels(L, part, I, I_seg, numpts, params, params2)


    X = zeros(2 * numpts, params.numscales * (params.nfeats + params2.nfeats));
    labels = zeros(2 * numpts, 1);
    p = []; n = []; t = [];
    for i = 1:size(I_seg, 3)
        if ~isempty(find(I_seg(:,:,i) == 1))
            p = [p; i];
        end

        z = I(:,:,i,1);
        if sum(z(:)) > 0
            n = [n; i];
        end

        if ~isempty(find(I_seg(:,:,i) == 2))
            t = [t; i];
        end

    end
    
    % First sample negatives
    for i = 1:numpts
        index = n(mod(i-1,length(n)) + 1);
        if strcmp(part,'tumor')
            if mod(i, 2) == 0
                [row, col] = find(I_seg(:,:,index) < 2 & I(:,:,index,1) > 0);
            else
                index = p(mod(i-1,length(p)) + 1);
                [row, col] = find(I_seg(:,:,index) == 1 & I(:,:,index,1) > 0);
            end
        elseif strcmp(part,'edema')
            [row, col] = find(I_seg(:,:,index) < 1 & I(:,:,index,1) > 0);
        end
        ind = randperm(length(row));
        row = row(ind); col = col(ind);         
        X(i,:) = L(row(1), col(1), index, :);
        labels(i) = 1;
    end

    % Next sample edema + tumor
    if strcmp(part,'edema')
        for i = 1:numpts
            index = p(mod(i-1,length(p)) + 1);
            [row, col] = find(I_seg(:,:,index) > 0);
            ind = randperm(length(row));
            row = row(ind); col = col(ind);
            X(numpts + i,:) = L(row(1), col(1), index, :);
            labels(numpts + i) = 2;
        end
    end

    % Next sample tumor
    if strcmp(part,'tumor')
        for i = 1:numpts
            index = t(mod(i-1,length(t)) + 1);
            [row, col] = find(I_seg(:,:,index) == 2);
            ind = randperm(length(row));
            row = row(ind); col = col(ind);
            X(numpts + i,:) = L(row(1), col(1), index, :);
            labels(numpts + i) = 2;
        end
    end

    % Permute
    ind = randperm(2*numpts);
    X = X(ind, :);
    labels = labels(ind, :);
