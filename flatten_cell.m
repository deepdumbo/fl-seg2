function C = flatten_cell(A)

    C = {}; 
    for i = 1:numel(A)  
        if(~iscell(A{i}))
            C = [C,A{i}];
        else
            Ctemp = flatten_cell(A{i});
            C = [C,Ctemp{:}];
        end
    end
