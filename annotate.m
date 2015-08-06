function yhat = annotate(image, model, mask, scaleparams)

    % Mask should be a binary image
    [m,n,p] = size(image);
    image = bsxfun(@times, image, mask);
    image = reshape(image, m * n, p);
    image = standard(image, scaleparams);
    
    %[~, yhat] = max(image * model, [], 2);
    
    [yhat, M] = predict(model, image);
    %[~,~,M] = run_data_through_network(model, image, [scaleparams.optval 0.5]);
    yhat = reshape(M(2,:), m, n, 1);
    
    %yhat = cellfun(@str2num, predict(model, image));
    
    %yhat = run_data_through_network(model, image, [0.5 0.5]);
    %yhat = reshape(yhat, m, n, 1) - 1;
    
end

