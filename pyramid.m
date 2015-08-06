function images = pyramid(I, params)

    images = cell(params.numscales * size(I,3), 1);
    %h = fspecial('gaussian');

    % Loop over images
    for i = 1:size(I, 3)
        
        % Build a pyramid
        %y = lpd(I(:,:,i), 'Burt', numscales);
        
        y = cell(params.numscales, 1);
        %y{1} = I(:,:,i);
        %for j = 2:params.numscales
        %    im = I(:,:,i);
        %    y{j} = imresize(im, 2^(-(j-1)));
        %end
        z = Gscale(I(:,:,i), params.numscales, [5 5], 1);
        for j = 1:params.numscales
            y{j} = z(j).img;
        end
             
        % Loop over layers
        for j = 1:length(y)
            yj = padarray(y{j}, [(params.rfSize(1) - 1) / 2 (params.rfSize(2) - 1) / 2], 'replicate');
            images{params.numscales*(i-1) + j} = yj;
        end
        
    end

end

