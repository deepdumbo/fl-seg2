function results = vessels(model, D, D2, params, params2, scaleparams)

    % Cell for results
    results = cell(20, 1);

    % Loop over each volume
    for i = 1:20

        disp(['Segmenting vessel ' num2str(i) ' of 20...']);

        % Load the volume
        [I, I_seg] = load_vessels(i);

        % Loop over slices
        Y = zeros(size(I), 'uint8');
        for j = 1:size(I, 3)

            % Compute the segmentation
            yhat = segment(I(:,:,j), I_seg(:,:,j), model, D, D2, params, params2, scaleparams);

            % Scale probabilities to [0,255]
            Y(:,:,j) = round(yhat * 255);

        end
        disp(' ');

        % Append the results
        results{i} = Y;

    end

    % Save the results
    save('vessel_results','results');
