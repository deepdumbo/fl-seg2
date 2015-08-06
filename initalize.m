function network = initalize(X, layers)

    % Initialize the network
    D = size(X, 2);
    no_layers = length(layers);
    numconn = 5;
    decay = 0.25;


    network{1}.bias_upW = zeros(1, layers(1)) + 0.1;

    network{1}.W = zeros(D, layers(1));
    for j = 1 : D
        idx = ceil(layers(1) * rand(1, numconn));
        network{1}.W(j, idx) = randn(numconn, 1);
    end
    network{1}.W = decay * network{1}.W;

    %r  = sqrt(6) / sqrt(D + layers(1) +1);
    %network{1}.W = rand(D, layers(1)) * 2 * r - r;

    %network{1}.W = randn(D, layers(1)) * 0.01;

    for i=2:no_layers
        %network{i}.W = randn(layers(i - 1), layers(i)) * .0001;
        network{i}.bias_upW = zeros(1, layers(i)) + 0.1;

        network{i}.W = zeros(layers(i-1), layers(i));
        for j = 1:layers(i-1)
            idx = ceil(layers(i) * rand(1, numconn));
            network{i}.W(j, idx) = randn(numconn, 1);
        end
        network{i}.W = decay * network{i}.W;


        %r  = sqrt(6) / sqrt(layers(i-1) + layers(i) + 1);
        %network{i}.W = rand(layers(i-1), layers(i)) * 2 * r - r;

        %network{i}.W = randn(layers(i-1), layers(i)) * 0.01;
    end
    %network{no_layers}.W = randn(layers(end), D) * .0001;
    %network{no_layers}.bias_upW = zeros(1, D);



