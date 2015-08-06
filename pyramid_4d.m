function images = pyramid_4d(I, params)

    images = cell(params.numscales * size(I,3), 1);
    %h = fspecial('gaussian');

    % Loop over images
    for i = 1:size(I, 3)

        x = cell(size(I,4), 1);
        
        % Build a pyramid
        %y = lpd(I(:,:,i), 'Burt', numscales);
        
        y = cell(params.numscales, 1);
        %y{1} = I(:,:,i);
        %for j = 2:params.numscales
        %    im = I(:,:,i);
        %    y{j} = imresize(im, 2^(-(j-1)));
        %end

        for j = 1:size(I,4)
            x{j} = Gscale(I(:,:,i,j), params.numscales, [5 5], 1);
        end
        for j = 1:params.numscales
            y{j} = cat(3, x{1}(j).img, x{2}(j).img, x{3}(j).img, x{4}(j).img);
        end
             
        % Loop over layers
        for j = 1:length(y)
            yj = padarray(y{j}, [(params.rfSize(1) - 1) / 2 (params.rfSize(2) - 1) / 2], 'replicate');
            images{params.numscales*(i-1) + j} = yj;
        end
        
    end

end
