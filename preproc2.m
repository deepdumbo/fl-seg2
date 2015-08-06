function L = preproc2(L, params2)

    % Loop through each image and pad
    for i = 1:size(L, 1)
        
        L{i} = padarray(L{i}, [(params2.rfSize(1) - 1) / 2 (params2.rfSize(2) - 1) / 2], 'replicate');
        
    end

end

