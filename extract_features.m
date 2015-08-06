function L = extract_features(X, D, params)


    % Parameters
    rfSize = params.rfSize;
    regSize = params.regSize;
    layer = params.layer;
    nmaps = params.nfeats;
    if layer == 1
        mapprod = 14;
    else
        mapprod = 5;
    end
    
    % Initalize
    %XC = zeros(size(X, 1), mapprod * nmaps);
    %prows = regSize(1) - rfSize(1) + 1;
    %pcols = regSize(2) - rfSize(2) + 1;
    %znargout = nargout;
    %lidimsize = params.lidimsize;
    %if znargout > 1
    L = cell(size(X, 1), 1);
    %end
    
    % Main Loop
    chunksize = 100;
    m = size(X, 1);
    numchunks = ceil(m ./ 100);
    
    % Extract features one 'chunk' at a time
    for chunk = 1:numchunks
        
        batch = X((chunk-1) * chunksize + 1 : min([chunk * chunksize end]));
        %XC_batch = zeros(size(chunk, 1), mapprod * nmaps);
        L_batch = cell(size(chunk, 1), 1);
    
        parfor i = 1:size(batch, 1)

            im = double(batch{i});
            im = squeeze(im);
            prows = size(im, 1) - rfSize(1) + 1;
            pcols = size(im, 2) - rfSize(2) + 1;

            % Extract subregions of the image
            [subregions, rowinds, colinds] = window(im, params);
            features = zeros(prows * pcols * nmaps, size(subregions, 1));
            
            % Compute spatial dimensions of next layer
            %if znargout > 1
            %    fudimsize = [prows * length(rowinds) pcols * length(colinds)];
            %    maxfudim = max(fudimsize);
            %    quot = round(maxfudim / params.maxS);
            %    lidimsize = [round(fudimsize(1) / quot) round(fudimsize(2) / quot)];
            %end
            
            % Extract subfeatures
            for j = 1:size(subregions, 1)     
                features(:, j) = extract_subfeatures(subregions(j,:), D, im, params);
            end

            % Reshape into spatial region
            index = 1;
            field = zeros(prows * length(rowinds), pcols * length(colinds), nmaps);
            for j = 1:length(rowinds)
                for k = 1:length(colinds)
                    field(prows*(j-1) + 1:prows*j, pcols*(k-1) + 1: pcols*k, :) = reshape(features(:, index), [prows pcols nmaps]);
                    index = index + 1;
                end
            end

            % Pooling
            %middle = spatial_pooling(field(:)', prows * length(rowinds), pcols * length(colinds), nmaps, [2 2]);
            %top = spatial_pooling(field(:)', prows * length(rowinds), pcols * length(colinds), nmaps, [1 1]);
            %if layer == 1
            %    bottom = spatial_pooling(field(:)', prows * length(rowinds), pcols * length(colinds), nmaps, [3 3]);
            %    XC_batch(i,:) = [bottom middle top];
            %else
            %    XC_batch(i,:) = [middle top];
            %end
            %if znargout > 1
            L_batch{i} = reshape(field, [prows * length(rowinds), pcols * length(colinds), nmaps]);
            %end
    
                    
        end

        %XC((chunk - 1) * chunksize + 1 : min([chunksize * chunk end]), :) = XC_batch;
        %if znargout > 1
        L((chunk - 1) * chunksize + 1 : min([chunksize * chunk end])) = L_batch;
        %end
        fprintf('.\n');
        
    end
    
    % Put L into cell format (if applicable)
    %if znargout > 1
        %L = reshape(L, [size(L,1) regSize(1) regSize(2) nmaps]);
        %L = mat2cell(L, ones(size(L, 1), 1), regSize(1), regSize(2),
        %nmaps);
    %end
    
    %disp(' ');
    



