from diffusers import AutoencoderKL
import torch
import torch.nn as nn
from torchvision.utils import save_image
from celeb_data import dataloader

url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoencoderKL.from_single_file(url)
        # self.model = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
      out = self.model.encode(image)
      out = out.latent_dist.sample()*0.18215
      out = (1/0.18215) * out
      output = self.model.decode(out)[0] 
      # print(output)
      return output
    def decoder(self,image):
      image = (1/0.18215)*image
      return self.model.decode(image)[0] 
    def latent(self,image):
      out = self.model.encode(image)
      out = out.latent_dist.sample()
      return out*0.18215
  

# vae = VAE().to('cuda').half()
# # # # # x = torch.randn(1,3,256,256)
# images,_ = next(iter(dataloader))
# images = images.to('cuda')
# # print(images.min(),images.max())
# latent = vae.latent(images)
# cons_image = vae.decoder(latent)
# print(latent.min(),latent.max())
# # # # # print(cons_image.shape)
# save_image(images,'saved_images\or.png')
# save_image(cons_image,'saved_images\img.png')