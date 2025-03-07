    %% Check if a GPU is available
    gpuAvailable = canUseGPU();
    
    %% Define the Discriminator
    % The discriminator expects flattened 28x28 images (784 pixels)
    discriminator = dlnetwork(layerGraph([
        featureInputLayer(784, 'Normalization', 'none', 'Name', 'input')
        fullyConnectedLayer(1024, 'Name', 'fc1')
        leakyReluLayer(0.2, 'Name', 'leaky1')
        dropoutLayer(0.5, 'Name', 'drop1')
        fullyConnectedLayer(512, 'Name', 'fc2')
        leakyReluLayer(0.2, 'Name', 'leaky2')
        dropoutLayer(0.4, 'Name', 'drop2')
        fullyConnectedLayer(256, 'Name', 'fc3')
        leakyReluLayer(0.2, 'Name', 'leaky3')
        dropoutLayer(0.4, 'Name', 'drop3')
        fullyConnectedLayer(1, 'Name', 'fc4')
        sigmoidLayer('Name', 'sigmoid')
    ]));
    
    %% Define the Generator
    % The generator produces a 784-element vector for each sample.
    generator = dlnetwork(layerGraph([
        featureInputLayer(100, 'Normalization', 'none', 'Name', 'input')
        fullyConnectedLayer(256, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(512, 'Name', 'fc2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(1024, 'Name', 'fc3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        fullyConnectedLayer(784, 'Name', 'fc4')
        tanhLayer('Name', 'tanh')
    ]));
    
    %% Load and Preprocess FashionMNIST Data
    % Set the paths to your FashionMNIST raw files (update these paths as needed):
    % imagesFile = 'C:\Users\jaywe\OneDrive - University of Plymouth\MSc EEE\Dissertation\FashionMNIST\raw\train-images-idx3-ubyte';
    imagesFile = 'C:\Users\jaywe\OneDrive - University of Plymouth\MSc EEE\Dissertation\MNIST Digit\train-images-idx3-ubyte';
    % Note: labels are not used for unconditional GAN training.
    trainImages = loadMNISTImages(imagesFile);  % Returns a [28,28,numImages] array
    numTrainImages = size(trainImages, 3);
    data = cell(numTrainImages, 1);
    
    % Preprocess each image:
    for i = 1:numTrainImages
        img = trainImages(:, :, i);      % 28x28 grayscale image
        img = double(img) / 255;           % Scale to [0, 1]
        img = 2 * img - 1;                 % Map [0,1] -> [-1, 1] (to match tanh)
        data{i} = reshape(img, 28*28, 1);   % Flatten to 784x1 column vector
    end
    
    % Concatenate all images to form XTrain (size: [784, NumImages])
    XTrain = cat(2, data{:});
    
    % Convert to a dlarray with format 'CB' (Channel, Batch)
    if canUseGPU()
        XTrain = gpuArray(dlarray(XTrain, 'CB'));
    else
        XTrain = dlarray(XTrain, 'CB');
    end
    
    %% Training Parameters
    numEpochs = 35;
    miniBatchSize = 128;
    numBatches = floor(size(XTrain, 2) / miniBatchSize);
    
    lrG = 0.0002;
    lrD = 0.0002;
    beta1 = 0.5;
    beta2 = 0.999;
    epsilon = 1e-8;
    iteration = 0;
    adamOptD = AdamOptimizer(lrD, beta1, beta2, epsilon);
    adamOptG = AdamOptimizer(lrG, beta1, beta2, epsilon);
    
    
    
    % Preallocate arrays to store average losses per epoch
    discriminator_loss_history = zeros(numEpochs,1);
    generator_loss_history = zeros(numEpochs,1);
    disp(numBatches)
    %% Training Loop
    for epoch = 1:numEpochs
        epochDLoss = 0;
        epochGLoss = 0;
        
        for batch = 1:numBatches
            % Select a mini-batch of real images (flattened: [784, miniBatchSize])
            idx = (batch-1)*miniBatchSize + 1 : batch*miniBatchSize;
            realImages = XTrain(:, idx);
            iteration = iteration + 1;
            
            % Generate fake images using the generator.
            noise = dlarray(randn(100, miniBatchSize, 'single'), 'CB');
            fakeImages = forward(generator, noise);
            
            % Create labels: 1 for real, 0 for fake.
            realLabels = ones(1, miniBatchSize, 'single');
            fakeLabels = zeros(1, miniBatchSize, 'single');
            
            % Concatenate images and labels along the batch dimension.
            combinedImages = cat(2, realImages, fakeImages);  % Size: [784, 2*miniBatchSize]
            combinedLabels = dlarray([realLabels, fakeLabels], 'CB');  % Size: [1, 2*miniBatchSize]
            
            % Evaluate the discriminator loss and gradients using dlfeval.
            [lossD, gradientsD] = dlfeval(@modelLossDiscriminator, discriminator, combinedImages, combinedLabels);
            
            % Evaluate the generator loss and gradients using dlfeval.
            noise = dlarray(randn(100, miniBatchSize, 'single'), 'CB');
            [lossG, gradientsG] = dlfeval(@modelLossGenerator, generator, discriminator, noise);
            
            % Update networks using Adam update rule.
            % After computing gradientsD and gradientsG via dlfeval...
            [discriminator] = adamOptD.step(discriminator, gradientsD);
            [generator] = adamOptG.step(generator, gradientsG);
    
            % Accumulate losses
            epochDLoss = epochDLoss + double(gather(extractdata(lossD)));
            epochGLoss = epochGLoss + double(gather(extractdata(lossG)));
        end
        
        % Compute average losses for this epoch
        avgDLoss = epochDLoss / numBatches;
        avgGLoss = epochGLoss / numBatches;
        discriminator_loss_history(epoch) = avgDLoss;
        generator_loss_history(epoch) = avgGLoss;
        
        % Display epoch number and average losses.
        disp(['Epoch: ' num2str(epoch) ' | Discriminator Loss: ' num2str(avgDLoss, '%.4f') ', Generator Loss: ' num2str(avgGLoss, '%.4f')]);
        %% Generate and Save Test Images in a Grid after each Epoch
        % Generate test images
        testNoise = dlarray(randn(100, 100, 'single'), 'CB');
        fakeImagesTest = forward(generator, testNoise);
        fakeImagesTest = gather(extractdata(fakeImagesTest));  % Size: [784, 100]
        
        % Create a figure showing a 10x10 grid of generated images
        figure('Visible', 'off');
        numRows = 10;
        numCols = 10;
        for k = 1:100
            subplot(numRows, numCols, k);
            img = reshape(fakeImagesTest(:, k), [28, 28]);
            img = (img + 1) / 2;  % Map from [-1,1] to [0,1]
            imshow(img, []);
            axis off;
        end
        sgtitle(sprintf('Epoch %d | D Loss: %.4f | G Loss: %.4f', epoch, avgDLoss, avgGLoss));
        drawnow;
        
        % Create folder for saving results if it doesn't exist
        resultFolder = 'gan_results';
        if ~exist(resultFolder, 'dir')
            mkdir(resultFolder)
        end
        % Save the figure as a PNG file
        filenamePNG = fullfile(resultFolder, sprintf('gan%03d.png', epoch));
        exportgraphics(gcf, filenamePNG, 'Resolution', 300);
        close(gcf);
    
    end
    
    
    %% Compile Saved PNG Frames into a GIF using FFmpeg
    gifFilename = fullfile(resultFolder, 'output.gif');
    % Replace with the full path to FFmpeg if not in your PATH
    ffmpegPath = 'C:\ffmpeg\bin\ffmpeg.exe';
    ffmpegCommand = sprintf('"%s" -y -framerate 2 -i %s/gan%%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" %s', ...
                            ffmpegPath, resultFolder, gifFilename);
    system(ffmpegCommand);
    
    %% Plot Loss Histories
    figure;
    plot(1:numEpochs, discriminator_loss_history, '-o', 'LineWidth', 2);
    hold on;
    plot(1:numEpochs, generator_loss_history, '-o', 'LineWidth', 2);
    xlabel('Epoch');
    ylabel('Loss');
    legend('Discriminator Loss', 'Generator Loss');
    title('Loss History');
    grid on;
    saveas(gcf, fullfile(resultFolder, 'loss-history.png'));
    close(gcf);
    
    %% Local Functions
    
    function [loss, gradients] = modelLossDiscriminator(discriminator, images, labels)
        % Compute predictions using the discriminator.
        predictions = forward(discriminator, images);
        epsilon = 1e-8;
        % Compute binary cross-entropy loss.
        loss = -mean(labels .* log(predictions + epsilon) + (1 - labels) .* log(1 - predictions + epsilon), 'all');
        % Compute gradients with respect to the learnable parameters.
        gradients = dlgradient(loss, discriminator.Learnables);
    end
    
    function [loss, gradients] = modelLossGenerator(generator, discriminator, noise)
        % Generate fake images using the generator.
        fakeImages = forward(generator, noise);
        % Compute predictions for fake images using the discriminator.
        predictions = forward(discriminator, fakeImages);
        epsilon = 1e-8;
        % Compute the generator loss.
        loss = -mean(log(predictions + epsilon), 'all');
        % Compute gradients with respect to the generator's learnable parameters.
        gradients = dlgradient(loss, generator.Learnables);
    end
    
    function [net, state] = adamupdate(net, gradients, state, iteration, learningRate, beta1, beta2, epsilon)
        if iteration == 1
            % Initialize state structure
            state.Velocity = cell(size(net.Learnables,1),1);
            state.SquaredGradient = cell(size(net.Learnables,1),1);
            for i = 1:size(net.Learnables,1)
                state.Velocity{i} = zeros(size(net.Learnables.Value{i}), 'like', net.Learnables.Value{i});
                state.SquaredGradient{i} = zeros(size(net.Learnables.Value{i}), 'like', net.Learnables.Value{i});
            end
        end
    
        for i = 1:size(net.Learnables,1)
            % Update biased first moment estimate.
            state.Velocity{i} = beta1 * state.Velocity{i} + (1 - beta1) * gradients.Value{i};
            % Update biased second raw moment estimate.
            state.SquaredGradient{i} = beta2 * state.SquaredGradient{i} + (1 - beta2) * gradients.Value{i}.^2;
            % Compute bias-corrected first moment estimate.
            vCorrected = state.Velocity{i} / (1 - beta1^iteration);
            % Compute bias-corrected second moment estimate.
            sCorrected = state.SquaredGradient{i} / (1 - beta2^iteration);
            % Update parameters.
            net.Learnables.Value{i} = net.Learnables.Value{i} - learningRate * vCorrected ./ (sqrt(sCorrected) + epsilon);
        end
    end
    
    function images = loadMNISTImages(filename)
        % Open the file in binary mode with big-endian ordering.
        fid = fopen(filename, 'rb');
        if fid == -1
            error('Could not open %s', filename);
        end
    
        magic = fread(fid, 1, 'int32', 0, 'ieee-be');
        if magic ~= 2051
            error('Invalid magic number in MNIST image file: %s', filename);
        end
    
        numImages = fread(fid, 1, 'int32', 0, 'ieee-be');
        numRows   = fread(fid, 1, 'int32', 0, 'ieee-be');
        numCols   = fread(fid, 1, 'int32', 0, 'ieee-be');
        
        % Read the remaining data as unsigned chars
        images = fread(fid, inf, 'unsigned char');
        fclose(fid);
        
        % Reshape into a 3D array [numRows, numCols, numImages]
        images = reshape(images, numCols, numRows, numImages);
        images = permute(images, [2, 1, 3]);  % Now [rows, cols, images]
    end
    
    function labels = loadMNISTLabels(filename)
        fid = fopen(filename, 'rb');
        if fid == -1
            error('Could not open %s', filename);
        end
    
        magic = fread(fid, 1, 'int32', 0, 'ieee-be');
        if magic ~= 2049
            error('Invalid magic number in MNIST label file: %s', filename);
        end
    
        numLabels = fread(fid, 1, 'int32', 0, 'ieee-be');
        labels = fread(fid, inf, 'unsigned char');
        fclose(fid);
    end
