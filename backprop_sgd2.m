function [network, results] = backprop_sgd2(network, train, train_labels, test, test_L, noise, lambda)


    % Initialize some variables
    numpts = size(train, 1);
    no_layers = length(network);
    deltaW = cell(1, no_layers);
    deltaB = cell(1, no_layers);
    for i = 1:no_layers
        deltaW{i} = zeros(size(network{i}.W));
        deltaB{i} = zeros(size(network{i}.bias_upW));
    end
    maxiter = 100;
    n = 100;
    verbose = floor(numpts ./ n);

    % Parameters
    eta_t = 10;
    f = 0.998;
    p_i = 0.5;
    p_f = 0.99;
    T = 500;
    L = 15;
    p_t = (1 / T) * p_i + (1 - (1 / T)) * p_f;

    % Pre-compute batches
    index = randperm(numpts);
    numbatches = ceil(numpts ./ n);
    batchX = cell(numbatches, 1);
    batchY = cell(numbatches, 1);
    for batch = 1:n:numpts
        batchX{batch} = train(index(batch:min([batch + n - 1 numpts])),:);
        batchY{batch} = train_labels(index(batch:min([batch + n - 1 numpts])),:);
    end

    % Main loop
    for iter = 1:maxiter

        % Display
        %disp([' - epoch ' num2str(iter) ' of ' num2str(maxiter) '...']);

        % Loop over all batches
        %index = randperm(numpts);
        loop = 1;
        for batch=1:n:numpts

            %fprintf('.');

            % Select current batch
            %X = train(index(batch:min([batch + n - 1 numpts])),:);
            %Y = train_labels(index(batch:min([batch + n - 1 numpts])),:);
            X = batchX{batch};
            Y = batchY{batch};
            sx = size(X, 1);

            % Corruption
            if noise(1) > 0
                X(rand(size(X)) < noise(1)) = 0;
            end

            % Run data through network
            acts = cell(1, no_layers + 1);
            drop = cell(1, no_layers + 1);
            acts{1} = [X ones(sx, 1)];
            for i=1:no_layers
                if i ~= no_layers
                    acts{i + 1} = [max(acts{i} * [network{i}.W; network{i}.bias_upW], 0) ones(sx, 1)];
                    if noise(2) > 0 && i == no_layers - 1
                        temp = acts{i+1}(:,1:end-1);
                        drop{i+1} = rand(size(temp)) < noise(2);
                        temp(drop{i+1}) = 0;
                        acts{i+1}(:,1:end-1) = temp;
                    end
                else
                    acts{i + 1} = [acts{i} * [network{i}.W; network{i}.bias_upW] ones(sx, 1)];
                end
            end
            M = acts{end}(:,1:end-1)';
            M = exp(bsxfun(@minus, M, max(M, [], 1)));
            M = bsxfun(@rdivide, M, sum(M));
 
            % Compute cost
            sqW = network{end}.W.^2;
            C = (-1 / sx) * sum(sum(Y' .* log(M))) + (lambda / 2) * sum(sqW(:));

            % Backprop
            dW = cell(1, no_layers);
            db = cell(1, no_layers);
            Ix = (-1 / sx) * (Y' - M)';

            for i=no_layers:-1:1

                % Compute update
                delta = acts{i}' * Ix;
                if i == no_layers
                    delta(1:end-1, :) = delta(1:end-1, :) + lambda * network{end}.W;
                end
                dW{i} = delta(1:end - 1,:);
                db{i} = delta(end,:);
            
                if i > 1
                    Ix = (Ix * [network{i}.W; network{i}.bias_upW]') .* (acts{i} > 0);
                    Ix = Ix(:,1:end - 1);
                end

                % Update delta
                deltaW{i} = p_t * deltaW{i} - (1 - p_t) * (eta_t ./ sx) * dW{i};
                deltaB{i} = p_t * deltaB{i} - (1 - p_t) * (eta_t ./ sx) * db{i};

                % Update weights
                network{i}.W = network{i}.W + deltaW{i};
                network{i}.bias_upW = network{i}.bias_upW + deltaB{i};

                % Re-scale weights (if necessary)
                colw = sum(network{i}.W.^2, 1) + eps;
                NW = repmat(colw > L, size(network{i}.W, 1), 1);
                NW = bsxfun(@times, NW, sqrt(L) ./ sqrt(colw));
                NW(NW == 0) = 1;
                network{i}.W = network{i}.W .* NW;

                % These lines are probably not important 
                colb = sum(network{i}.bias_upW.^2) + eps;
                NB = repmat(colb > L, size(network{i}.bias_upW, 1), 1);
                NB = bsxfun(@times, NB, sqrt(L) ./ sqrt(colb));
                NB(NB == 0) = 1;
                network{i}.bias_upW = network{i}.bias_upW .* NB;
            
            end

            % Get test error
            %if mod(loop, verbose) == 0
            %    yhat = run_data_through_network(network, test, noise);
            %    %disp(['Kappa: ' num2str(quadraticWeightedKappa(test_L, yhat))]);
            %    disp(['Accuracy: ' num2str(sum(test_L == yhat) ./ length(yhat))]);
            %end
            loop = loop + 1;
        end
 
        % Update parameters
        eta_t = eta_t * f;
        if iter < T
            p_t = ((iter + 1) / T) * p_i + (1 - ((iter + 1) / T)) * p_f;
        else
            p_t = p_f;
        end

        % Get error
        %yhat = run_data_through_network(network, train, noise);
        %[foo, train_L] = max(train_labels, [], 2);
        %results.train_error(iter) = mean(train_L ~= yhat);
        %disp(['Error: ' num2str(results.train_error(iter))]);

        %yhat = run_data_through_network(network, test, noise);
        %results.test_error(iter) = mean(test_L ~= yhat);
        %disp(['Accuracy: ' num2str(1 - results.test_error(iter))]);

    end





   

