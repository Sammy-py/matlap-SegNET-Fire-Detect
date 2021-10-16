classdef ElementWiseMultiplication < nnet.layer.Layer
    % Example custom ElementWiseMultiplication layer.
    properties (Learnable)
        % Layer learnable parameters
            
        % Scaling coefficients
    end
    
    methods
        function layer = ElementWiseMultiplication(numInputs,name) 
            % layer = ElementWiseMultiplication(numInputs,name) creates a
            % element wise multiplication and specifies the number of inputs
            % and the layer name.
            % Set number of inputs.
            layer.NumInputs = numInputs;
            % Set layer name.
            layer.Name = name;
            % Set layer description.
            layer.Description = "Element Wise Multiplication of " + numInputs +  ... 
                " inputs";
        
        end
        
        function Z = predict(~, X1,X2)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.     
            % Element Wise Multiplication
                        Z = X1 .*X2;
        
        end
    end
end