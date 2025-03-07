classdef ReshapeLayer < nnet.layer.Layer
    properties
        % Target size (excluding the batch dimension)
        TargetSize
    end
    properties
        % Declare that this layer outputs formatted data.
        OutputFormats = {'SSCB'};
    end
    methods
        function layer = ReshapeLayer(targetSize, name)
            % Constructor: store target size and set name/description.
            layer.Name = name;
            layer.Description = "Reshape layer to " + join(string(targetSize), 'x');
            layer.TargetSize = targetSize;
        end
        
        function Z = predict(layer, X)
            % Here X is a formatted dlarray with format 'SSCB'
            % Expected size of X is [7*7*128, 1, 1, Batch]
            batchSize = size(X,4);
            % Extract the underlying data.
            dataX = extractdata(X);
            % (Optionally, verify that dataX is of size [7*7*128, 1, 1, Batch].)
            % Reshape the data into [7, 7, 128, Batch]
            reshapedData = reshape(dataX, [layer.TargetSize, batchSize]);
            % Return a formatted dlarray.
            Z = dlarray(reshapedData, 'SSCB');
        end
        
        function Z = forward(layer, X)
            Z = layer.predict(X);
        end
    end
end
