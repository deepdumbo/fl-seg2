function [yhat, mappedX, M] = run_data_through_network(network, X, noise)
%RUN_DATA_THROUGH_NETWORK Run data through the network
%
%   mappedX = run_data_through_network(network, X)
%
% Runs the dataset X through the parametric t-SNE embedding defined in
% network. The result is returned in mappedX.
%

% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology


    % Run the data through the network
    n = size(X, 1);
    mappedX = [X ones(n, 1)];
    for i=1:length(network) - 1
        if i ~= 1
            mappedX = [max(mappedX * [network{i}.W .* (1 - 0); network{i}.bias_upW], 0) ones(n, 1)];
            %mappedX = [tanh(mappedX * [network{i}.W .* 0.5; network{i}.bias_upW]) ones(n, 1)];
            %mappedX = [1 ./ (1 + exp(-(mappedX * [network{i}.W .* 0.5; network{i}.bias_upW]))) ones(n, 1)];
        else
            mappedX = [max(mappedX * [network{i}.W .* (1 - noise(1)); network{i}.bias_upW], 0) ones(n, 1)];
            %mappedX = [tanh(mappedX * [network{i}.W; network{i}.bias_upW]) ones(n, 1)];
            %mappedX = [1 ./ (1 + exp(-(mappedX * [network{i}.W; network{i}.bias_upW]))) ones(n, 1)];
        end
    end

    % Softmax
    M = mappedX * [network{end}.W .* (1 - noise(2)); network{end}.bias_upW];
    M = M';
    M = exp(bsxfun(@minus, M, max(M, [], 1)));
    M = bsxfun(@rdivide, M, sum(M));

    % Logistic
    %yhat = [1 ./ (1 + exp(-(mappedX * [network{end}.W .* 0.5; network{end}.bias_upW]))) ones(n, 1)];
    %M = yhat(:,1:end-1);
    %M = M';

    % Get argmax
    [foo,yhat] = max(M, [], 1);
    yhat = yhat';

    
