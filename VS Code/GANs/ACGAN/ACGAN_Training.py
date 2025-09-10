import torchvision
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt 

from ACGAN import ACDiscriminator, ACGenerator

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
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)

test_ds = LatentImageDataset(
    latents_npy   = f"D:/EEGImageNet/latent_dumps/Time/{g}/test_latentsNew.npy",
    filenames_npy = f"D:/EEGImageNet/latent_dumps/Time/{g}/test_filenamesNew.npy",
    labels_npy    = f"D:/EEGImageNet/latent_dumps/Time/{g}/test_stringLabelsNew.npy",
    image_root    = f"D:/EEGImageNet/imageNet_images",
    transform     = img_transform,
    subject       = f"D:/EEGImageNet/latent_dumps/Time/{g}/test_subjectNew.npy",
)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=False)

val_latents = np.load(f"D:/EEGImageNet/latent_dumps/Time/{g}/val_latentsNew.npy")       # (N_val, 128)
val_labels  = np.load(f"D:/EEGImageNet/latent_dumps/Time/{g}/val_stringLabelsNew.npy")  # (N_val,), dtype=str
val_filenames = np.load(f"D:/EEGImageNet/latent_dumps/Time/{g}/val_filenamesNew.npy")  # (N_val,)
val_subject = np.load(f"D:/EEGImageNet/latent_dumps/Time/{g}/val_subjectNew.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path    = "GANs/ACGAN/checkpointFine0"
if not os.path.exists(save_path):
    os.makedirs("checkpoint", exist_ok=True)

# Create folder to hold result
result_folder = 'ACGAN-result'
if not os.path.exists(result_folder ):
    os.makedirs(result_folder )

with open(os.path.join("GANs", "capsGAN", "capsGAN-result", "fixed_val_filenames.txt")) as f:
    fixed_filenames = [ln.strip() for ln in f if ln.strip()]

fname_to_idx = {fn: i for i, fn in enumerate(val_filenames)}
fixed_idxs   = [fname_to_idx[fn] for fn in fixed_filenames]

fixed_latents = torch.from_numpy(val_latents[fixed_idxs]).float().to(device)
cls2idx = train_ds.cls2idx
fixed_label_strs = [val_labels[i] for i in fixed_idxs]
fixed_label_idxs = [cls2idx[s]           for s in fixed_label_strs]
fixed_labels     = torch.tensor(
    fixed_label_idxs, dtype=torch.long, device=device
)

# instantiate AC-GAN
n_classes = 8
generator     = ACGenerator(latent_dim=128, n_classes=n_classes).to(device)
discriminator = ACDiscriminator(n_classes=n_classes).to(device)

# losses
adversarial_loss = nn.BCEWithLogitsLoss()
auxiliary_loss   = nn.CrossEntropyLoss()

# Total loss history for both networks
D_loss_history = []
G_loss_history = []

# optimisers
lr = 2e-4
optimiserG = optim.Adam(generator.parameters(),    lr=lr,    betas=(0.5, 0.999))
optimiserD = optim.Adam(discriminator.parameters(),lr=lr/2,    betas=(0.5, 0.999))

ckpt = torch.load(os.path.join(save_path,"acGANcheckpoint_e1000.pth"))
generator.load_state_dict(ckpt["generator_state"])
discriminator.load_state_dict(ckpt["discriminator_state"])
epoch_start     = ckpt['epoch'] + 1
optimiserG.load_state_dict(ckpt['optimiserG_state'])
optimiserD.load_state_dict(ckpt['optimiserD_state'])
D_loss_history, G_loss_history = ckpt['D_loss_history'],ckpt['G_loss_history']

LAMBDA_C = 2.0

# training loop
num_epochs = 2000
for epoch in range(epoch_start, num_epochs+1):
    D_loss_epoch = 0.0
    G_loss_epoch = 0.0

    #Put both networks into training mode
    generator.train()
    discriminator.train()

    for z, real_imgs, labels in train_loader:
        B = real_imgs.size(0)
        z          = z.to(device)
        real_imgs  = real_imgs.to(device)
        labels     = labels.to(device)

        # True and False Labels.  128 is the batch size
        true_label = torch.empty(B,1,device=device).uniform_(0.8,1.0)
        false_label = torch.empty(B,1,device=device).uniform_(0.0,0.2)

        # ── Train Discriminator ──
        # real
        validity_real, pred_cls_real = discriminator(real_imgs)
        lossD_real_adv = adversarial_loss(validity_real, true_label)
        lossD_real_cls = auxiliary_loss(pred_cls_real, labels)

        # fake
        fake_imgs = generator(z, labels).detach()
        validity_fake, pred_cls_fake = discriminator(fake_imgs)
        lossD_fake_adv = adversarial_loss(validity_fake, false_label)

        #Total loss and optimister step
        lossD = lossD_real_adv + lossD_fake_adv + lossD_real_cls

        optimiserD.zero_grad()
        lossD.backward()
        optimiserD.step()

        # ── Train Generator ──

        gen_imgs = generator(z, labels)
        validity, pred_cls = discriminator(gen_imgs)
        # want discriminator to label these as “real”
        lossG_adv = adversarial_loss(validity, true_label)
        # want discriminator to predict the correct class
        lossG_cls = auxiliary_loss(pred_cls, labels)

        lossG = lossG_adv + LAMBDA_C*lossG_cls

        optimiserG.zero_grad()
        lossG.backward()
        optimiserG.step()

        D_loss_epoch += lossD.item()
        G_loss_epoch += lossG.item()

    print(f"[Epoch {epoch}/{num_epochs}] "
          f"D_loss: {D_loss_epoch/len(train_loader):.4f}, "
          f"G_loss: {G_loss_epoch/len(train_loader):.4f}")

    # averaging
    avg_D = D_loss_epoch / len(train_loader)
    avg_G = G_loss_epoch / len(train_loader)
    D_loss_history.append(avg_D)
    G_loss_history.append(avg_G)
    
    # validation grid:
    if epoch % 5 == 0:
        with torch.no_grad():
            generator.eval()
            fake = generator(fixed_latents, fixed_labels)    # now pass labels too
            grid = torchvision.utils.make_grid(
                fake,
                nrow=10,
                normalize=True,
                value_range=(-1,1)
            )
            np_grid = grid.cpu().permute(1,2,0).numpy()
            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(np_grid)
            ax.axis("off")
            ax.set_title(
                f"Epoch {epoch}/{num_epochs}\n"
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

    # checkpointing:
    if epoch % 25 == 0 or avg_G < min(G_loss_history):
        torch.save({
            "epoch": epoch,
            "generator_state": generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
            "optimiserG_state": optimiserG.state_dict(),
            "optimiserD_state": optimiserD.state_dict(),
            "D_loss_history": D_loss_history,
            "G_loss_history": G_loss_history,
        }, os.path.join(save_path, f"acGANcheckpoint_e{epoch:03d}.pth"))
        print(f"✓ checkpoint saved at epoch {epoch}")

plt.clf()
plt.plot(D_loss_history, label="Discriminator")
plt.plot(G_loss_history, label="Generator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(result_folder, "Network_Loss_History.png"))