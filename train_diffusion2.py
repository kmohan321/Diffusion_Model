# from unet_complete import UNET
from diffusion import diffusion
import torch.optim as optim
import torch
import torch.nn as nn 
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
from torchvision.utils import save_image
from celeb_data import dataloader
from torch.cuda.amp import GradScaler
from text_encoder import CLIPTextEncoder,BERT
from vae_pre import VAE
import time
from torchsummary import summary
from unetnew import UNET
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
from diffusers import UNet2DConditionModel
from torchsummary import summary
# model = UNet2DConditionModel(sample_size=16,in_channels=4,out_channels=4,cross_attention_dim=312,encoder_hid_dim=312,down_block_types=("DownBlock2D","CrossAttnDownBlock2D",\
#   "CrossAttnDownBlock2D"),up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D","UpBlock2D"),block_out_channels=(64,128,256)).to(device)
unet = UNET(in_channels=4,out_channels=4,embedd_length=1280,
prompt_embed_length=312,down_block_types=("CROSSATTNDOWNBLOCK2D",
    "CROSSATTNDOWNBLOCK2D","DOWNBLOCK2D"),down_block_channels=(128,256),
    up_block_types=("UPBLOCK2D","CROSSATTNUPBLOCK2D","CROSSATTNUPBLOCK2DSPECIAL"),num_heads=8).to(device)
# summary(model)
file_path = 'saved_images\images_{}.png'
load_model_unet = False
epochs = 1000
lr = 1e-4

scaler = GradScaler()
diff = diffusion(device)
bert_encoder = BERT(device)
if load_model_unet==True:
  model.load_state_dict(torch.load('unet_model.pth'))
vae = VAE().to(device)
optim_unet = optim.Adam(unet.parameters(),lr=lr)
loss_fn = nn.MSELoss(reduction='none')


global_steps = 0
for epoch in range(1,epochs+1):
  unet.train()
  start_time = time.time()
  for i ,(images,texts) in tqdm(enumerate(dataloader)):
      global_steps += 1
      with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        images = images.to(device)
        latent = vae.latent(images)
        noise_images,noise,timesteps = diff.noised_images(latent)
        timesteps = timesteps.to(device)
        text_encodings = bert_encoder(texts)
        
        pred_noise = unet(noise_images,timesteps.squeeze(1),text_encodings)
        loss  = loss_fn(pred_noise,noise)
        loss = loss.reshape(loss.shape[0],-1).sum(dim=1)
        loss = loss.mean()

      optim_unet.zero_grad()
      scaler.scale(loss).backward()
      scaler.unscale_(optim_unet)
      torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
      scaler.step(optim_unet)
      scaler.update()
      print(f"epoch {epoch}|batch {i}/{len(dataloader)}| unet_loss {loss:.4f}")
  
  end_time = time.time()
  print(end_time-start_time)
  torch.save(unet.state_dict(),'unet_model.pth')
  with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    sampled_img = diff.sampling(unet,text_encodings)
    with torch.no_grad():
      sampled_img = vae.decoder(sampled_img)
  save_image(sampled_img,file_path.format(epoch))



# code for testing
# images,texts = next(iter(dataloader))
# for epoch in range(1,epochs+1):
#       model.train()
#       with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
#           images = images.to(device)
#           text_encodings = bert_encoder(texts)
#           latent = vae.latent(images)
#           noise_images,noise,timesteps = diff.noised_images(latent)
#           timesteps = timesteps.to(device)
#           pred_noise = model(noise_images,timesteps.squeeze(1),text_encodings)
#         #   print(pred_noise.min(),pred_noise.max())
#           loss  = loss_fn(pred_noise,noise)
#           loss = loss.reshape(loss.shape[0],-1).sum(dim=1)
#           loss = loss.mean()
#       optim_unet.zero_grad()
#       loss.backward()
#       torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#       optim_unet.step()
#       # writer.add_graph(model,[noise_images,timesteps.squeeze(1),text_encodings])
#       print(f" epoch {epoch} | loss {loss}")
# with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
#   sampled_img = diff.sampling(model,text_encodings)
#   with torch.no_grad():
#     sampled_img = vae.decoder(sampled_img)
# save_image(sampled_img,file_path.format(epoch+1))






