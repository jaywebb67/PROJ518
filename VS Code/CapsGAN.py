
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import itertools
import matplotlib.pyplot as plt 
import numpy as np

import subprocess
import os

##############################################################################
# Simple squash function for the “Reshape + squash” step
##############################################################################
def squash(x, dim=-1, eps=1e-8):
    """
    Squash activation commonly used in Capsule Networks.
    x shape: (batch, ..., vector_dim)
    Returns a tensor of the same shape after non-linear 'squash'.
    """
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)  # ||x||^2
    scale = squared_norm / (1.0 + squared_norm)
    scale = scale / torch.sqrt(squared_norm + eps)
    return scale * x

##############################################################################
# A simple “Capsule” layer in fully connected form: Linear -> reshape -> softmax
##############################################################################
class CapsuleLayerFC(nn.Module):
    """
    Transforms an incoming vector of size `in_dim` to `num_caps` capsules,
    each of dimension `dim_caps`. Then applies softmax over `num_caps`
    and flattens back to (batch, num_caps * dim_caps).
    
    E.g., in_dim=9216, num_caps=10, dim_caps=16 => output shape = 160
    """
    def __init__(self, in_dim, num_caps=10, dim_caps=16):
        super().__init__()
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.linear = nn.Linear(in_dim, num_caps * dim_caps)

    def forward(self, x):
        """
        x shape: (batch, in_dim)
        output shape: (batch, num_caps * dim_caps), i.e. (batch, 160) if 10x16
        """
        # 1) Linear transform
        out = self.linear(x)  # (batch, 10*16)

        # 2) Reshape to (batch, 10, 16)
        out = out.view(-1, self.num_caps, self.dim_caps)

        # 3) Softmax over the capsule axis (num_caps=10)
        out = F.softmax(out, dim=1)

        # 4) Flatten back to (batch, 160)
        out = out.view(-1, self.num_caps * self.dim_caps)
        return out

##############################################################################
# Discriminator model matching the table’s layer-by-layer specification
##############################################################################
class CapsGANDiscriminator(nn.Module):
    """
    Implements the Discriminator architecture given in the table:
    
    1)  Input: 28 x 28 x 1
        Conv2D+LeakyReLU (kernel=9, out_channels=256) -> (20 x 20 x 256)
    2)  BatchNorm2d(0.8) -> (20 x 20 x 256)
    3)  Primary capsule (another Conv2D 9x9, out_channels=256, stride=2)
        -> (6 x 6 x 256)
    4)  Reshape + squash -> (1152 x 8)
    5)  BatchNorm1d(0.8) -> (1152 x 8)
    6)  Flatten -> 9216
    7)  Capsule layer1 + Softmax -> (10x16=160)
    8)  Dense+LeakyReLU -> 160
    9)  Capsule layer2 + Softmax -> 160
    10) Dense+LeakyReLU -> 160
    11) Capsule layer3 + Softmax -> 160
    12) Dense+LeakyReLU -> 160
    13) Dense+Sigmoid -> 1
    """

    def __init__(self):
        super().__init__()

        # 1) Conv2D + LeakyReLU => out: (256, 20, 20)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, 
                               kernel_size=9, stride=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        # 2) BatchNorm2d with momentum=0.8
        self.bn2d_1 = nn.BatchNorm2d(256, momentum=0.8)

        # 3) "Primary capsule" (2nd conv) => out: (256, 6, 6)
        #    Table says: kernel=9x9, out_channels=256, we want 6x6 => use stride=2
        self.primary_caps = nn.Conv2d(in_channels=256, out_channels=256,
                                      kernel_size=9, stride=2)
        
        # 4) We will do the "reshape + squash" in forward()
        #    shape => (batch, 256, 6, 6) => 9216 => (1152 x 8) => squash

        # 5) After squash, do BatchNorm1d(0.8). We'll apply it to the dimension=8
        #    so we'll do BN on the last dimension. We'll handle the shape in forward.
        self.bn1d_caps = nn.BatchNorm1d(num_features=8, momentum=0.8)

        # 6) Flatten => from (1152 x 8) to 9216
        #    We'll do the actual flatten in forward(). No dedicated module needed.

        # 7) Capsule layer1 => (10x16=160)
        self.caps1 = CapsuleLayerFC(in_dim=9216, num_caps=10, dim_caps=16)

        # 8) Dense+LeakyReLU => 160
        self.dense1 = nn.Sequential(
            nn.Linear(160, 160),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 9) Capsule layer2 => 160
        self.caps2 = CapsuleLayerFC(in_dim=160, num_caps=10, dim_caps=16)

        # 10) Dense+LeakyReLU => 160
        self.dense2 = nn.Sequential(
            nn.Linear(160, 160),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 11) Capsule layer3 => 160
        self.caps3 = CapsuleLayerFC(in_dim=160, num_caps=10, dim_caps=16)

        # 12) Dense+LeakyReLU => 160
        self.dense3 = nn.Sequential(
            nn.Linear(160, 160),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 13) Dense+Sigmoid => 1
        self.out = nn.Sequential(
            nn.Linear(160, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x shape: (batch, 1, 28, 28)
        returns: (batch, 1)
        """
        # 1) Conv2D + LeakyReLU
        x = self.conv1(x)               # (batch, 256, 20, 20)
        x = self.leaky_relu1(x)

        # 2) BatchNorm2d
        x = self.bn2d_1(x)             # still (batch, 256, 20, 20)

        # 3) Primary capsule (2nd conv)
        x = self.primary_caps(x)       # (batch, 256, 6, 6)

        # 4) Reshape + squash => (batch, 1152, 8)
        #    256 * 6 * 6 = 9216. 9216 / 8 = 1152.
        batch_size = x.size(0)
        x = x.view(batch_size, 256*6*6)         # (batch, 9216)
        x = x.view(batch_size, 1152, 8)         # (batch, 1152, 8)
        x = squash(x, dim=2)                    # squash along last dim => (batch, 1152, 8)

        # 5) BatchNorm1d => shape is (batch, 1152, 8).
        #    By default, nn.BatchNorm1d expects (batch, C, sequence_length) or (batch, C).
        #    We can transpose so that "8" is the channel dimension:
        x = x.transpose(1, 2)                   # (batch, 8, 1152)
        x = self.bn1d_caps(x)                   # batchnorm on 8 channels
        x = x.transpose(1, 2)                   # back to (batch, 1152, 8)

        # 6) Flatten => (batch, 9216)
        x = x.reshape(batch_size, -1)

        # 7) Capsule layer1 + Softmax => (batch, 160)
        x = self.caps1(x)

        # 8) Dense + LeakyReLU => 160
        x = self.dense1(x)

        # 9) Capsule layer2 + Softmax => 160
        x = self.caps2(x)

        # 10) Dense + LeakyReLU => 160
        x = self.dense2(x)

        # 11) Capsule layer3 + Softmax => 160
        x = self.caps3(x)

        # 12) Dense + LeakyReLU => 160
        x = self.dense3(x)

        # 13) Dense + Sigmoid => 1
        x = self.out(x)  # (batch, 1)

        return x
   
class Generator(nn.Module):
    """
    Generator model matching the layer sequence:

    (1) 100 (noise)        -> Dense+ReLU (128*7*7=6272) -> shape=(batch, 6272)
    (2) Reshape            -> (batch, 128, 7, 7)
    (3) BatchNorm2d(0.8)   -> (batch, 128, 7, 7)
    (4) UpSampling2D       -> (batch, 128, 14, 14)
    (5) Conv2D+ReLU        -> (batch, 128, 14, 14)
    (6) BatchNorm2d(0.8)   -> (batch, 128, 14, 14)
    (7) UpSampling2D       -> (batch, 128, 28, 28)
    (8) Conv2D+ReLU        -> (batch, 64, 28, 28)
    (9) BatchNorm2d(0.8)   -> (batch, 64, 28, 28)
    (10) Conv2D+Tanh       -> (batch, 1, 28, 28)
    """
    def __init__(self, latent_dim=100):
        super().__init__()

        # 1) Dense + ReLU => 128*7*7 = 6272
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(inplace=True)
        )

        # 2) We reshape in forward() to (batch, 128, 7, 7)

        # 3) BatchNorm2d(128, momentum=0.8)
        self.bn_reshape = nn.BatchNorm2d(128, momentum=0.8)

        # 4) UpSampling2D => (14 x 14)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")

        # 5) Conv2D + ReLU => (128 channels)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, 
                               kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        # 6) BatchNorm2d(128, momentum=0.8)
        self.bn1 = nn.BatchNorm2d(128, momentum=0.8)

        # 7) UpSampling2D => (28 x 28)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")

        # 8) Conv2D + ReLU => 64 channels
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, 
                               kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # 9) BatchNorm2d(64, momentum=0.8)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.8)

        # 10) Conv2D + Tanh => 1 channel (28 x 28 x 1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, 
                               kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        x shape: (batch, 100) noise
        Returns: (batch, 1, 28, 28) generated image
        """
        # 1) Dense + ReLU => (batch, 6272)
        x = self.fc(x)

        # 2) Reshape => (batch, 128, 7, 7)
        x = x.view(x.size(0), 128, 7, 7)

        # 3) BatchNorm2d => (batch, 128, 7, 7)
        x = self.bn_reshape(x)

        # 4) UpSampling => (14,14)
        x = self.upsample1(x)

        # 5) Conv2D + ReLU => (batch, 128, 14, 14)
        x = self.conv1(x)
        x = self.relu1(x)

        # 6) BatchNorm2d => (batch, 128, 14, 14)
        x = self.bn1(x)

        # 7) UpSampling => (28,28)
        x = self.upsample2(x)

        # 8) Conv2D + ReLU => (batch, 64, 28, 28)
        x = self.conv2(x)
        x = self.relu2(x)

        # 9) BatchNorm2d => (batch, 64, 28, 28)
        x = self.bn2(x)

        # 10) Conv2D + Tanh => (batch, 1, 28, 28)
        x = self.conv3(x)
        x = self.tanh(x)

        return x

def noise(data_size):
    '''
    Generates data_size number of random noise
    '''
    noise_features = 100
    n = torch.randn(data_size, noise_features)
    return n

def im_convert(tensor):
    '''
        Convert Tensor to displable format
    '''
    image = tensor.to("cpu").clone().detach()
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)

    return image

def margin_loss(pred, target, m_plus=0.9, m_minus=0.1, lambda_=0.5):
    """
    pred:   (batch, 1)  The Discriminator's scalar output for each example
    target: (batch, 1)  1.0 for real, 0.0 for fake

    Returns a scalar mean loss based on:
       L = T * max(0, m+ - pred)^2 + lambda_ * (1 - T) * max(0, pred - m-)^2
    """
    # Flatten to (batch,)
    pred = pred.view(-1)
    target = target.view(-1)

    # Real-label part: T * (max(0, m+ - v))^2
    L_real = target * F.relu(m_plus - pred).pow(2)
    # Fake-label part: (1 - T) * (max(0, v - m-))^2
    L_fake = (1.0 - target) * F.relu(pred - m_minus).pow(2)

    return (L_real + lambda_ * L_fake).mean()

# Loading training dataset

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
    ])

dataset = datasets.FashionMNIST(root='dataset/', train=True, 
                        transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=128, 
                        drop_last=True,
                        shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Create discriminator and generator
discriminator = CapsGANDiscriminator().to(device)
generator = Generator(latent_dim=100).to(device)

# Create 100 test_noise for visualizing how well our model perform.
test_noise = noise(100).to(device)

# Optimizers and loss
lr = 0.0002
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.MarginRankingLoss()

# True and False Labels.  128 is the batch size
true_label = torch.ones(128, 1).to(device)
false_label = torch.zeros(128, 1).to(device)

# Create folder to hold result
result_folder = 'gan1-result'
if not os.path.exists(result_folder ):
    os.makedirs(result_folder )

# Training in action
print("Starting Training...")

num_epochs = 35
discriminator_loss_history = []
generator_loss_history = []

for epoch in range(1, num_epochs+1):
    discriminator_batch_loss = 0.0
    generator_batch_loss = 0.0
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        batch_size = data.shape[0]

        discriminator.zero_grad()

        # Train discriminator to get better at differentiate real/fake data
        # 1.1 Train discriminator on real data
        d_real_predict = discriminator(data)
        d_real_loss = criterion(input1 = d_real_predict, target = true_label)
  

        # 1.2 Train discriminator on fake data from generator
        d_fake_noise = noise(batch_size).to(device)
        # Generate outputs and detach to avoid training the Generator on these labels
        d_fake_input = generator(d_fake_noise).detach()
        d_fake_predict = discriminator(d_fake_input)
        d_fake_loss = criterion(d_fake_predict, false_label)

        # 1.3 combine real loss and fake loss for discriminator
        discriminator_loss = d_real_loss + d_fake_loss
        discriminator_batch_loss += discriminator_loss.item()
        discriminator_loss.backward()
        optimizerD.step()


        # Train generator to get better at deceiving discriminator
        g_fake_noise = noise(batch_size).to(device)
        g_fake_input = generator(g_fake_noise)
        generator.zero_grad()
        # Get prediction from discriminator
        g_fake_predict = discriminator(g_fake_input)
        generator_loss = criterion(g_fake_predict, true_label)
        generator_batch_loss += generator_loss.item()
        generator_loss.backward()
        optimizerG.step()

        # print loss every 100 batches
        if (batch_idx + 1) % 100 == 0:

            print(f'Epoch [{epoch}/{num_epochs}]  Batch {batch_idx + 1}/{len(dataloader)} \
                    Loss D: {discriminator_loss:.4f}, Loss G: {generator_loss:.4f}')


    discriminator_loss_history.append(discriminator_batch_loss / (batch_idx + 1))
    generator_loss_history.append(generator_batch_loss / (batch_idx + 1))

    with torch.no_grad():
        
        fake_images = generator(test_noise)

        size_figure_grid = 10
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(10*10):
            i = k // 10
            j = k % 10
            ax[i, j].cla()
            ax[i, j].imshow(im_convert(fake_images[k].view(1,28,28)))

        label = 'Epoch {0}'.format(epoch)
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig(result_folder + "/gan%03d.png" % epoch)
        plt.show(block=False)
        plt.pause(1.5)
        plt.close(fig)


# create gif, 2 frames per second
subprocess.call([
    'ffmpeg', '-framerate', '2', '-i', \
    result_folder + '/gan%03d.png', result_folder+'/output.gif'
])


plt.clf()
plt.plot(discriminator_loss_history, label='discriminator loss')
plt.plot(generator_loss_history, label='generator loss')
plt.savefig(result_folder + "/loss-history.png")
plt.legend()
plt.show()


