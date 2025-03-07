function Y = squashCapsules(X)
% squashCapsules - Applies the squash nonlinearity to capsule vectors.
%
%   Y = squashCapsules(X) takes a dlarray X of size 
%       [H, W, numCaps, dimCaps, N]
%   and applies the squash function defined as:
%
%       v_j = (||s_j||^2 / (1 + ||s_j||^2)) * (s_j / ||s_j||)
%
%   along the capsule dimension (dimCaps). A small epsilon is used for numerical stability.

    epsilon = 1e-8;
    
    % Compute squared norm along dimension 4 (capsule dimension)
    squaredNorm = sum(X.^2, 4);
    
    % Compute the norm, adding epsilon to avoid division by zero
    normVal = sqrt(squaredNorm + epsilon);
    
    % Compute the scale factor for each capsule:
    % scale = (||s_j||^2 / (1 + ||s_j||^2)) / ||s_j||
    scale = squaredNorm ./ (1 + squaredNorm) ./ normVal;
    
    % Expand scale along the capsule dimension.
    % Use repmat to match the size of X along the 4th dimension.
    scale = repmat(scale, [1, 1, 1, size(X,4), 1]);
    
    % Multiply the input by the scale factor.
    Y = X .* scale;
end
