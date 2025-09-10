import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

# from capsGAN import CapsGANDiscriminator,CapsGANGenerator,margin_loss
# from GAN import GANGenerator,GANDiscriminator
from DCGAN import DCGANGenerator, DCGANDiscriminator

warnings.filterwarnings("ignore", "Truncated File Read", UserWarning)




class LatentImageDataset(Dataset):
    def __init__(self, latents_npy, filenames_npy, labels_npy,image_root,subject, transform):
        # load 128-D latents and N filenames
        self.latents  = np.load(latents_npy)          # (N,128)
        self.filenames = np.load(filenames_npy,allow_pickle=True)       # (N,), dtype=str
        self.labels = np.load(labels_npy)
        self.image_root = image_root                  
        self.transform  = transform
        self.sub        = subject
        # build mapping from string label → integer in [0,80)
        classes = np.unique(self.labels)
        self.cls2idx = {c:i for i,c in enumerate(classes)}

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        # get latent vector
        z = torch.from_numpy(self.latents[idx]).float()
        # load the RGB stimulus image
        cls = self.labels[idx]
        path = os.path.join(self.image_root, cls, self.filenames[idx])
        img  = Image.open(path).convert("RGB")         # 3 channels
        img  = self.transform(img)                    # → Tensor 3×28×28
        # integer label
        y = torch.tensor(self.cls2idx[cls], dtype=torch.long)
        return z, img,y


# transforms & dataloaders
img_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

g = "fine0"

train_ds = LatentImageDataset(
    latents_npy   = f"D:/EEGImageNet/latent_dumps/Time/{g}/train_latentsNew.npy",
    filenames_npy = f"D:/EEGImageNet/latent_dumps/Time/{g}/train_filenamesNew.npy",
    labels_npy    = f"D:/EEGImageNet/latent_dumps/Time/{g}/train_stringLabelsNew.npy",
    image_root    = f"D:/EEGImageNet/imageNet_images",
    subject       = f"D:/EEGImageNet/latent_dumps/Time/{g}/train_subjectNew.npy",
    transform     = img_transform,
)
train_dataloader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)

test_ds = LatentImageDataset(
    latents_npy   = f"D:/EEGImageNet/latent_dumps/Time/{g}/test_latentsNew.npy",
    filenames_npy = f"D:/EEGImageNet/latent_dumps/Time/{g}/test_filenamesNew.npy",
    labels_npy    = f"D:/EEGImageNet/latent_dumps/Time/{g}/test_stringLabelsNew.npy",
    image_root    = f"D:/EEGImageNet/imageNet_images",
    transform     = img_transform,
    subject       = f"D:/EEGImageNet/latent_dumps/Time/{g}/test_subjectNew.npy",
)
test_dataloader = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=False)

val_latents = np.load(f"D:/EEGImageNet/latent_dumps/Time/{g}/val_latentsNew.npy")       # (N_val, 128)
val_labels  = np.load(f"D:/EEGImageNet/latent_dumps/Time/{g}/val_stringLabelsNew.npy")  # (N_val,), dtype=str
val_filenames = np.load(f"D:/EEGImageNet/latent_dumps/Time/{g}/val_filenamesNew.npy")  # (N_val,)
val_subject = np.load(f"D:/EEGImageNet/latent_dumps/Time/{g}/val_subjectNew.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path    = os.path.join("GANs", "DCGAN", "DCGAN-checkpoint",f"{g}")
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Create folder to hold result
result_folder = os.path.join("GANs", "DCGAN", "DCGAN-result",f"{g}New")
if not os.path.exists(result_folder ):
    os.makedirs(result_folder )

with open(os.path.join("GANs", "capsGAN", "capsGAN-result", "fixed_val_filenames.txt")) as f:
    fixed_filenames = [ln.strip() for ln in f if ln.strip()]

fname_to_idx = {fn: i for i, fn in enumerate(val_filenames)}
fixed_idxs   = [fname_to_idx[fn] for fn in fixed_filenames]

fixed_latents = torch.from_numpy(val_latents[fixed_idxs]).float().to(device)

# Create discriminator and generator
discriminator = DCGANDiscriminator().to(device)
generator = DCGANGenerator(latent_dim=128).to(device)

# Training in action
num_epochs = 1000

# Learning rate
lr = 2e-4

#optimisers 
optimiserG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimiserD = optim.Adam(discriminator.parameters(), lr=lr/2, betas=(0.5, 0.999))

D_loss_history = []
G_loss_history = []

# ckpt = torch.load("/content/drive/MyDrive/ImageNet_Images/GANs - Fine 2/capsGAN/checkpoint/capsGANcheckpoint_e500.pth")
# generator.load_state_dict(ckpt["model"])
# discriminator.load_state_dict(ckpt["disc"])
# epoch_start     = ckpt['epoch'] + 1
# optimiserG.load_state_dict(ckpt['optG'])
# optimiserD.load_state_dict(ckpt['optD'])
# D_loss_history, G_loss_history = ckpt['losses']


# before training loop

criterion = nn.BCELoss()


print("Starting Training…")
for epoch in range(1, num_epochs+1):
    D_loss_epoch, G_loss_epoch = 0., 0.
    discriminator.train()
    generator.train()
    for latents, real_imgs,_ in train_dataloader:
        latents   = latents.to(device)
        real_imgs = real_imgs.to(device)
        B = real_imgs.size(0)

        # True and False Labels.  128 is the batch size
        true_label = torch.empty(B,1,device=device).uniform_(0.8,1.0)
        false_label = torch.empty(B,1,device=device).uniform_(0.0,0.2)

        # ——— Discriminator step ———
        d_real = discriminator(real_imgs)
        loss_real = criterion(d_real, true_label)
        fake_imgs = generator(latents).detach()
        d_fake    = discriminator(fake_imgs)
        loss_fake = criterion(d_fake, false_label[:B])
        lossD     = loss_real + loss_fake
        
        optimiserD.zero_grad()
        lossD.backward() 
        optimiserD.step()
        D_loss_epoch += lossD.item()

        # ——— Generator step ———
        fake_imgs = generator(latents)
        d_out     = discriminator(fake_imgs)
        lossG     = criterion(d_out, true_label[:B])
        optimiserG.zero_grad()       
        lossG.backward()
        optimiserG.step()
        G_loss_epoch += lossG.item()

    # compute averages
    avg_D = D_loss_epoch / len(train_dataloader)
    avg_G = G_loss_epoch / len(train_dataloader)

    # append to history for plotting later
    D_loss_history.append(avg_D)
    G_loss_history.append(avg_G)


    print(f"Epoch {epoch}/{num_epochs}  "
          f"D_loss: {avg_D:.4f}, G_loss: {avg_G:.4f}")

    if epoch % 5 == 0:
        with torch.no_grad():
            generator.eval()
            fake = generator(fixed_latents)  # (80,3,28,28)
            grid = torchvision.utils.make_grid(
                fake,
                nrow=4,
                normalize=True,
                value_range=(-1,1)
            )

            # after you’ve built `grid` (a tensor C×H×W):
            np_grid = grid.cpu().permute(1,2,0).numpy()  # H×W×C in [0,1]

            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(np_grid)
            ax.axis("off")
            ax.set_title(
                f"DCGAN Epoch {epoch}/{num_epochs}\n"
                f"D_loss: {avg_D:.4f}, G_loss: {avg_G:.4f}",
                fontsize=12,
                color="white",
                backgroundcolor="black",
                pad=10
            )
            plt.tight_layout()
            plt.savefig(os.path.join(result_folder, f"epoch{epoch:03d}_valgrid.png"))
            plt.close(fig)
            generator.train()
    if epoch % 25 == 0 or avg_G < min(G_loss_history):
        # at save time (e.g. after epoch N):
        torch.save({
            "epoch":   epoch,
            "model":   generator.state_dict(),
            "disc":    discriminator.state_dict(),
            "optG":    optimiserG.state_dict(),
            "optD":    optimiserD.state_dict(),
            "losses":  (D_loss_history, G_loss_history),
        }, os.path.join(save_path,f"DCGANcheckpoint_epoch{epoch:03d}.pth"))
        print(f"✓ checkpoint saved at epoch {epoch}")

plt.clf()
plt.plot(D_loss_history, label="Discriminator")
plt.plot(G_loss_history, label="Generator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig(os.path.join(result_folder, "Network_Loss_History.png"))