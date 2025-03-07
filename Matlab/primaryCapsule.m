function Y = primaryCapsule(X, targetH, targetW, numFilters)
    % Ensure X is a dlarray with labels; if not, assign 'SSCB'

    if canUseGPU()
        X= gpuArray(dlarray(X, 'SSCB'));
    else
        X= dlarray(X, 'SSCB');
    end

    % Use imresize on the underlying data as a workaround.
    resizedData = imresize(extractdata(X), [targetH, targetW]);
    % Convert back to dlarray with appropriate labels.
    Y = dlarray(resizedData, 'SSCB');
    
    % Now Y is of size [targetH, targetW, numFilters, N]
    % Reshape to group channels into capsules.
    Y = reshape(Y, targetH, targetW, 32, 8, size(Y,4));
    
    % Apply the squash function.
    Y = squashCapsules(Y);
end
