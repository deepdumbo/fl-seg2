function [trainP, testP, scaleparams] = bts(test_volumes, class, grade, part, params, params2)


    % Get the indices for the training volumes
    if strcmp(grade, 'LG')
        train_volumes = setdiff([1 2 4 6 8 11 12 13 14 15], test_volumes);
    else
        train_volumes = setdiff(1:15, test_volumes);
    end
    %train_volumes = [2 3 4 5 6 7 8];

    % Learn the filters
    disp('Learning filters...');
    [D, D2] = learn_features_bts(train_volumes, class, grade, params, params2);

    % Extract features
    disp('Extracting features...');
    trainF = compute_features_bts(train_volumes, class, grade, D, D2, params, params2);
    testF = compute_features_bts(test_volumes, class, grade, D, D2, params, params2);

    % Sample points for logistic regression
    disp('Sampling points...');
    [train, train_labels] = sample_bts(train_volumes, class, part, grade, trainF, 2000, params, params2);
    [test, test_labels] = sample_bts(test_volumes, class, part, grade, testF, 2000, params, params2);

    % Scale the data
    [trainX, scaleparams] = standard(train);
    testX = standard(test, scaleparams);

    % Parameter search
    disp('Parameter search...');
    %vals = [2^3, 2^2, 2^1, 2^0, 2^-1, 2^-2, 2^-3, 2^-4, 2^-5, 2^-6, 2^-7, 2^-8, 2^-9, 2^-10, 2^-11];
    vals = [0];
    %results = zeros(length(vals), 1);
    %models = cell(length(vals), 1);
    %for i = 1:length(vals)
    %    models{i} = softmax_regression(trainX, train_labels, 2, vals(i));
    %    yhat = predict(models{i}, testX);
    %    results(i) = mean(yhat == test_labels);
    %end
    %[best, ind] = max(results);
    %scaleparams.results = results;
    %scaleparams.vals = vals;
    %disp(['Accuracy: ' num2str(best) ' with parameter ' num2str(vals(ind))]);

    [optval, acc] = xval_svm(train, train_labels, vals, 5); % Ignore the name, not an SVM
    scaleparams.optval = optval;
    model = initalize(trainX, [100 100 2]);
    tl = full(sparse(train_labels, 1:length(train_labels), 1))';
    model = backprop_sgd2(model, trainX, tl, [], [], [optval 0.5], 2e-5);
    
    % Output the probabilities for each volume
    trainP = cell(length(train_volumes), 1);
    testP = cell(length(test_volumes), 1);
    for i = 1:length(trainP)
        V = load_braintumor(class, grade, train_volumes(i));
        for j = 1:size(trainF{i}, 3)
            trainP{i}(:,:,j) = annotate(squeeze(trainF{i}(:,:,j,:)), model, squeeze(V(:,:,j,1)) > 0, scaleparams);
        end
    end
    for i = 1:length(testP)
        V = load_braintumor(class, grade, test_volumes(i));
        for j = 1:size(testF{i}, 3)
            testP{i}(:,:,j) = annotate(squeeze(testF{i}(:,:,j,:)), model, squeeze(V(:,:,j,1)) > 0, scaleparams);
        end
    end





    
