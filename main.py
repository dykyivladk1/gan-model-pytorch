import os
import random
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision.utils as vutils
from torch import optim
import matplotlib.animation as animation
from IPython.display import HTML
import time


manual_seed = 999
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.use_deterministic_algorithms(True)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
class CustomDataset(Dataset):
    def __init__(self,dir,transform = None):
        super().__init__()
        self.dir = dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f))]

    def __len__(self):
        return len(self.image_filenames)
    def __getitem__(self,idx):
        img_path = os.path.join(self.dir,self.image_filenames[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

dataset = CustomDataset(dir = "archive/dataset/dataset", transform=transform)
dataloader = DataLoader(dataset,batch_size=128,shuffle=True)

print(len(dataloader))
class Gen(nn.Module):
    def __init__(self, ngpu):
        super(Gen, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self,input):
        return self.main(input)

class Dis(nn.Module):
    def __init__(self,ngpu):
        super(Dis,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3,64,4,2,1,bias=True),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64,64*2,4,2,1,bias=True),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64*2,64*4,4,2,1,bias=True),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64*4,64*8,4,2,1,bias=True),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64*8,1,4,1,0,bias=True),
            nn.Sigmoid()


            )

    def forward(self,input):
        return self.main(input)

ngpu = 1
device = torch.device("mps")

print(device)
netG = Gen(ngpu).to(device)
netD = Dis(ngpu).to(device)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(netG.parameters(),lr = 0.0002,betas=(0.5,0.999))
optimizer_D = torch.optim.Adam(netD.parameters(),lr = 0.0002,betas=(0.5,0.999))


fixed_noise = torch.randn(64,100,1,1,device=device)
real_label = 1
fake_label = 0


img_list = []

num_epochs = 200
iters = 0
print("Flying to MARS...")
start_time = time.time()
epoch_time_list = []

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    for i, data in enumerate(dataloader, 0):

        netD.zero_grad()
        real_cpu = data.to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, dtype=torch.float32, device=device)
        output = netD(real_cpu).view(-1)
        real_loss_d = criterion(output, label)
        real_loss_d.backward()

        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        fake_loss_d = criterion(output, label)
        fake_loss_d.backward()
        optimizer_D.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        loss_g = criterion(output, label)
        loss_g.backward()
        optimizer_G.step()

        if i == len(dataloader) - 1:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                fake_data = fake[0].numpy()
                fake_data = np.transpose(fake_data, (1, 2, 0))
                fake_data = ((fake_data + 1) * 255 / (2.0)).astype(np.uint8)


                img = Image.fromarray(fake_data)
                img.save(f'fake_image_epoch_{epoch}.jpg')
        iters += 1
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    if epoch == "50":
      torch.save(netG, 'generator_model_full.pth')
      torch.save(netD, 'discriminator_model_full.pth')

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    epoch_time_list.append(epoch_time)

    avg_epoch_time = sum(epoch_time_list) / len(epoch_time_list)
    remaining_time = avg_epoch_time * (num_epochs - epoch - 1)

    print(f"Epoch: {epoch}, Batch: {i}, Discriminator Real Loss: {real_loss_d.item()}, Discriminator Fake Loss: {fake_loss_d.item()}, Generator Loss: {loss_g.item()}")
    print(f"Time taken for this epoch: {epoch_time:.2f} seconds")
    print(f"Estimated time till end of training: {remaining_time:.2f} seconds")

end_time = time.time()
total_time = end_time - start_time

from torchvision.transforms import ToPILImage

to_pil = ToPILImage()


first_single_image_tensor = img_list[0][0].cpu().squeeze()
last_single_image_tensor = img_list[-1][0].cpu().squeeze()

first_single_image_tensor = (first_single_image_tensor / 2 + 0.5)
last_single_image_tensor = (last_single_image_tensor / 2 + 0.5)

first_image_pil = to_pil(first_single_image_tensor)
last_image_pil = to_pil(last_single_image_tensor)


first_image_pil_rgb = first_image_pil.convert("RGB")
last_image_pil_rgb = last_image_pil.convert("RGB")


first_image_pil_rgb.save('first_single_generated_image.png')
last_image_pil_rgb.save('last_single_generated_image.png')


final_img = np.transpose(img_list[-1],(1,2,0))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.imshow(final_img)
plt.savefig('final_generated_image.png')
plt.show()

torch.save(netG, 'generator_model_full.pth')
torch.save(netD, 'discriminator_model_full.pth')