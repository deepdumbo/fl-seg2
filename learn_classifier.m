function [model, scaleparams] = learn_classifier(X, labels, numfolds)

    % Values to search over
    %vals = [2^-7 2^-6 2^-5 2^-4 2^-3 2^-2 2^-1 2^0 2^1 2^2 2^3 2^4 2^5 2^6 2^7];
    vals = [2^0, 2^-1, 2^-2, 2^-3, 2^-4, 2^-5, 2^-6, 2^-7, 2^-8, 2^-9];
    %vals = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75];
    
    % First permute the data
    indperm = randperm(size(X, 1));
    X = X(indperm,:);
    labels = labels(indperm);

    % Apply cross-validation
    disp('Performing cross validation...')
    [optval, acc] = xval_svm(X, labels, vals, numfolds);
    disp(['Accuracy: ' num2str(max(acc) * 100) '%']);

    % Scale the data and train
    disp('Training Logistic Regression...')
    [X, scaleparams] = standard(X);
    scaleparams.optval = optval;
    %model = train_svm(X, labels, optval);
    model = softmax_regression(X, labels, 2, optval);
    %model = initalize(X, 2);
    %labels = full(sparse(labels, 1:length(labels), 1))';
    %model = backprop_sgd2(model, X, labels, [], [], [0.5 0.5], 5e-4);
