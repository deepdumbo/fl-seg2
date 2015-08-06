function D = dictionary(patches, params)


    % Parameters
    nfeats = params.nfeats;
    rfSize = params.rfSize;
    
    % Apply ZCA whitening
    %disp('Applying Whitening...');
    %if params.layer == 1
    %    D = whiten2(patches, params);
    %else
    %    D = whiten2(patches, params);
    %end
    %nX = bsxfun(@minus, patches, D.mean) * D.whiten; 
    D.mean = mean(patches);
    nX = bsxfun(@minus, patches, D.mean);
    
    % Train dictionary
    disp('Training Dictionary...');
    D.codes = run_omp1(nX, nfeats, 50);
    
    %params.data = nX';
    %params.Tdata = 5;
    %params.dictsize = nfeats;
    %params.iternum = 30;
    %params.memusage = 'high';
    %D.codes = ksvd(params,'i')';
    
end

