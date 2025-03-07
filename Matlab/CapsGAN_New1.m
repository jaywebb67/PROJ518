%% Optimized CapsGAN Implementation in MATLAB with GPU Support (with Labels)
% This script defines a generator and a capsule‐based discriminator,
% then trains a GAN on MNIST data using margin loss.
% (Adapt the loadMNISTImages function if you wish to use FashionMNIST.)

%% Data Loading & Preprocessing
imagesFile = 'C:\Users\jaywe\OneDrive - University of Plymouth\MSc EEE\Dissertation\MNIST Digit\train-images-idx3-ubyte';  % update with your MNIST images file path
trainImages = loadMNISTImages(imagesFile);  % [28,28,numImages]
numTrainImages = size(trainImages, 3);
data = cell(numTrainImages, 1);

for i = 1:numTrainImages
    img = double(trainImages(:, :, i)) / 255; % Scale to [0,1]
    img = 2 * img - 1;                       % Map to [-1,1]
    data{i} = reshape(img, [28,28,1]);        % Keep as 28x28x1
end

XTrain = cat(4, data{:});  % [28,28,1, NumImages]
if canUseGPU()
    XTrain = gpuArray(XTrain);
end

%% Generator Network Definition
generatorLG = layerGraph();
layersGen = [
    featureInputLayer(100, 'Name', 'noise')
    fullyConnectedLayer(128*7*7, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    functionLayer(@(X) reshapeGenerator(X), 'Name','reshape','Formattable',true)
    batchNormalizationLayer('Name','batchnorm1')
    resize2dLayer('Scale',2,'Method','bilinear','Name','upsample1')
    convolution2dLayer(3,128,'Padding','same','Name','conv1')
    reluLayer('Name','relu2')
    batchNormalizationLayer('Name','batchnorm2')
    resize2dLayer('Scale',2,'Method','bilinear','Name','upsample2')
    convolution2dLayer(3,64,'Padding','same','Name','conv2')
    reluLayer('Name','relu3')
    batchNormalizationLayer('Name','batchnorm3')
    convolution2dLayer(3,1,'Padding','same','Name','conv_final')
    tanhLayer('Name','tanh_output')
    ];
generatorLG = addLayers(generatorLG, layersGen);
executorG = dlnetwork(generatorLG);

%% Discriminator Network Definition (Capsule-based)
% This network uses a primary capsule layer implemented via functionLayers.
discriminatorLG = layerGraph();
layersDisc = [
    imageInputLayer([28,28,1], 'Name', 'input','Normalization','none')
    convolution2dLayer(9,256,'Stride',1,'Padding',0,'Name','conv1')
    leakyReluLayer(0.2,'Name','lrelu1')
    batchNormalizationLayer('Name','batchnorm1')
    
    % Primary Capsule Layer
    convolution2dLayer(9,256,'Stride',2,'Padding',0,'Name','primary_caps')
    functionLayer(@(X) reshapeCapsule(X), 'Name','reshape_caps','Formattable',true)
    functionLayer(@(X) squashFunction(X), 'Name','squash')
    batchNormalizationLayer('Name','batchnorm2')
    
    fullyConnectedLayer(9216,'Name','fc_caps')
    fullyConnectedLayer(160,'Name','caps1')
    
    % Final classification: single scalar for real/fake
    fullyConnectedLayer(1, 'Name', 'outputConnectedLayer')
    sigmoidLayer('Name', 'sigmoid_output')
    ];
discriminatorLG = addLayers(discriminatorLG, layersDisc);
executorD = dlnetwork(discriminatorLG);

%% Training Setup
numEpochs = 35;
miniBatchSize = 128;
lr = 0.0002;
% (You may adjust separate learning rates if desired.)
beta1 = 0.5;
beta2 = 0.999;
epsilon = 1e-8;
numBatches = floor(size(XTrain,4) / miniBatchSize);

% Create Adam optimizer objects (make sure an AdamOptimizer class/function is available)
adamOptD = AdamOptimizer(lr, beta1, beta2, epsilon);
adamOptG = AdamOptimizer(lr, beta1, beta2, epsilon);

% Preallocate loss history
discriminator_loss_history = zeros(numEpochs,1);
generator_loss_history = zeros(numEpochs,1);

disp(['Total mini-batches per epoch: ' num2str(numBatches)]);

%% Training Loop
for epoch = 1:numEpochs
    epochDLoss = 0;
    epochGLoss = 0;
    
    for batch = 1:numBatches
        % Get real images and convert to dlarray with SSCB format
        idx = (batch-1)*miniBatchSize + 1 : batch*miniBatchSize;
        realImages = dlarray(XTrain(:,:,:,idx),'SSCB');
        
        % Sample noise for generator input (feature dimension = 100)
        noiseBatch = dlarray(randn(100, miniBatchSize, 'single'),'CB');
        if canUseGPU(), noiseBatch = gpuArray(noiseBatch); end
        
        % Generate fake images from noise
        fakeImages = forward(executorG, noiseBatch);
        
        % Prepare labels: real=1, fake=0
        realLabels = ones(1, miniBatchSize, 'single');
        fakeLabels = zeros(1, miniBatchSize, 'single');
        % For discriminator training, combine real and fake data
        combinedImages = cat(4, realImages, fakeImages);
        combinedLabels = dlarray([realLabels, fakeLabels], 'CB');
        
        % Update Discriminator
        [dLoss, gradientsD] = dlfeval(@modelLossDiscriminator, executorD, combinedImages, combinedLabels);
        executorD = adamOptD.step(executorD, gradientsD);
        
        % Update Generator (wants its fakes to be labeled as real)
        [gLoss, gradientsG] = dlfeval(@modelLossGenerator, executorG, executorD, noiseBatch);
        executorG = adamOptG.step(executorG, gradientsG);
        
        % Accumulate losses
        epochDLoss = epochDLoss + double(gather(extractdata(dLoss)));
        epochGLoss = epochGLoss + double(gather(extractdata(gLoss)));
        
        % Optionally display mini-batch progress every 100 batches
        if mod(batch,100)==0
            fprintf('Epoch [%d/%d] Batch [%d/%d]  Loss D: %.4f, Loss G: %.4f\n', ...
                epoch, numEpochs, batch, numBatches, double(gather(extractdata(dLoss))), double(gather(extractdata(gLoss))));
        end
    end
    
    avgDLoss = epochDLoss / numBatches;
    avgGLoss = epochGLoss / numBatches;
    discriminator_loss_history(epoch) = avgDLoss;
    generator_loss_history(epoch) = avgGLoss;
    
    fprintf('Epoch %d | D Loss: %.4f | G Loss: %.4f\n', epoch, avgDLoss, avgGLoss);
    
    % Visualization every 5 epochs
    if mod(epoch,5)==0
        % Generate test images from fixed noise
        testNoise = dlarray(randn(100, miniBatchSize, 'single'),'CB');
        if canUseGPU(), testNoise = gpuArray(testNoise); end
        fakeImagesTest = forward(executorG, testNoise);
        fakeImagesTest = gather(extractdata(fakeImagesTest));
        
        % Create a figure grid to display 10x10 generated images
        figure('Visible','off');
        numRows = 10; numCols = 10;
        for k = 1:min(100, size(fakeImagesTest,4))
            subplot(numRows, numCols, k);
            img = squeeze(fakeImagesTest(:,:,:,k));
            img = (img + 1)/2; % Map from [-1,1] to [0,1]
            imshow(img,[]);
            axis off;
        end
        sgtitle(sprintf('Epoch %d',epoch));
        drawnow;
        
        resultFolder = 'capsgan_results';
        if ~exist(resultFolder, 'dir'), mkdir(resultFolder); end
        filenamePNG = fullfile(resultFolder, sprintf('capsgan_epoch%03d.png', epoch));
        exportgraphics(gcf, filenamePNG, 'Resolution', 300);
        close(gcf);
    end
end

trainedGAN = executorG;
disp('Training complete.');

%% ------------------ SUPPORTING LOSS FUNCTIONS ------------------

function [dLoss, gradientsD] = modelLossDiscriminator(executorD, combinedImages, combinedLabels)
    % Forward pass through discriminator
    predictions = forward(executorD, combinedImages);
    dLoss = marginLoss(predictions, combinedLabels);
    gradientsD = dlgradient(dLoss, executorD.Learnables);
end

function [gLoss, gradientsG] = modelLossGenerator(executorG, executorD, noise)
    generatedImages = forward(executorG, noise);
    predictions = forward(executorD, generatedImages);
    % Generator’s goal: have discriminator label fakes as real (label=1)
    onesLabels = ones(size(predictions), 'like', predictions);
    gLoss = marginLoss(predictions, onesLabels);
    gradientsG = dlgradient(gLoss, executorG.Learnables);
end

function lossVal = marginLoss(predictions, labels)
    % predictions should be in [0,1] (after sigmoid)
    mPlus = 0.9;
    mMinus = 0.1;
    lambdaVal = 0.5;
    
    termPos = labels .* max(0, mPlus - predictions).^2;       % for real images
    termNeg = (1 - labels) .* max(0, predictions - mMinus).^2;  % for fake images
    
    lossVal = mean(termPos + lambdaVal * termNeg, 'all');
end

%% ------------------ SUPPORTING FUNCTIONS ------------------
% Reshape function for the generator (from fc output to 7x7x128 feature map)
function Xout = reshapeGenerator(X)
    % X is of size [128*7*7, miniBatchSize]
    miniBatchSize = size(X,2);
    X = reshape(X, [7,7,128, miniBatchSize]);
    % Return as a dlarray in SSCB format
    Xout = dlarray(X, 'SSCB');
end

% Reshape function for the primary capsule layer in the discriminator.
% It reshapes the convolutional output into [1152, 8, 1, miniBatchSize]
function Xout = reshapeCapsule(X)
    % X is of size [H, W, C, miniBatchSize] where H and W are expected to be 6.
    % Here, 256*6*6 = 9216, and we reshape to [1152, 8, 1, miniBatchSize].
    miniBatchSize = size(X,4);
    X = reshape(X, [256*6*6, miniBatchSize]);
    X = reshape(X, [1152, 8, 1, miniBatchSize]);
    Xout = dlarray(X, 'SSCB');
end

% Squash function used in capsule networks.
function Xout = squashFunction(X)
    % X is a dlarray; we assume the capsule dimension is along the 2nd dimension.
    % Compute the norm along dimension 2.
    normX = sqrt(sum(X.^2, 2) + eps);
    % Squash function as described in Capsule Network literature.
    scale = (normX.^2) ./ (1 + normX.^2);
    Xout = scale .* (X ./ (normX + eps));
end

% --- Utility functions to load MNIST images (and labels if needed) ---
function images = loadMNISTImages(filename)
    fid = fopen(filename,'rb');
    if fid == -1, error('Could not open %s', filename); end
    magic = fread(fid,1,'int32',0,'ieee-be');
    if magic ~= 2051, error('Invalid magic number in MNIST image file: %s', filename); end
    numImages = fread(fid,1,'int32',0,'ieee-be');
    numRows   = fread(fid,1,'int32',0,'ieee-be');
    numCols   = fread(fid,1,'int32',0,'ieee-be');
    images = fread(fid,inf,'unsigned char');
    fclose(fid);
    
    images = reshape(images, numCols, numRows, numImages);
    images = permute(images, [2,1,3]);
end

function labels = loadMNISTLabels(filename)
    fid = fopen(filename,'rb');
    if fid == -1, error('Could not open %s', filename); end
    magic = fread(fid,1,'int32',0,'ieee-be');
    if magic ~= 2049, error('Invalid magic number in MNIST label file: %s', filename); end
    numLabels = fread(fid,1,'int32',0,'ieee-be');
    labels = fread(fid,inf,'unsigned char');
    fclose(fid);
end
