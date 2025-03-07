classdef AdamOptimizer
    % AdamOptimizer implements a simple version of the Adam optimizer for
    % updating the learnable parameters of a dlnetwork in a custom training loop.
    
    properties
        lr          % Learning rate (scalar)
        beta1       % Exponential decay rate for first moment estimates
        beta2       % Exponential decay rate for second moment estimates
        epsilon     % Small constant for numerical stability
        iteration   % Counter for number of updates
        state       % Structure holding optimizer state (velocity and squared gradients)
    end
    
    methods
        function obj = AdamOptimizer(lr, beta1, beta2, epsilon)
            % Constructor: Set hyperparameters.
            if nargin < 4
                epsilon = 1e-8;
            end
            obj.lr = lr;
            obj.beta1 = beta1;
            obj.beta2 = beta2;
            obj.epsilon = epsilon;
            obj.iteration = 0;
            obj.state = struct();
        end
        
        function net = step(obj, net, gradients)
            % STEP performs one Adam update on the dlnetwork 'net' using
            % the table 'gradients' (with the same format as net.Learnables).
            %
            % net: a dlnetwork object.
            % gradients: a table with the same dimensions as net.Learnables.
            
            obj.iteration = obj.iteration + 1;
            numParams = size(net.Learnables, 1);
            
            % Initialize state on the first iteration.
            if obj.iteration == 1
                obj.state.Velocity = cell(numParams, 1);
                obj.state.SquaredGradient = cell(numParams, 1);
                for i = 1:numParams
                    paramSize = size(net.Learnables.Value{i});
                    % Use the same data type and device as the parameter.
                    obj.state.Velocity{i} = zeros(paramSize, 'like', net.Learnables.Value{i});
                    obj.state.SquaredGradient{i} = zeros(paramSize, 'like', net.Learnables.Value{i});
                end
            end
            
            % Loop over each learnable parameter and update it.
            for i = 1:numParams
                grad = gradients.Value{i};
                
                % Update biased first moment estimate.
                obj.state.Velocity{i} = obj.beta1 * obj.state.Velocity{i} + (1 - obj.beta1) * grad;
                % Update biased second moment estimate.
                obj.state.SquaredGradient{i} = obj.beta2 * obj.state.SquaredGradient{i} + (1 - obj.beta2) * grad.^2;
                
                % Compute bias-corrected estimates.
                vCorrected = obj.state.Velocity{i} / (1 - obj.beta1^obj.iteration);
                sCorrected = obj.state.SquaredGradient{i} / (1 - obj.beta2^obj.iteration);
                
                % Update the parameter.
                net.Learnables.Value{i} = net.Learnables.Value{i} - obj.lr * vCorrected ./ (sqrt(sCorrected) + obj.epsilon);
            end
        end
    end
end

