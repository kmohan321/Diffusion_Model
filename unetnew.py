import torch
import torch.nn as nn
import torchsummary as summary
import math

# takes a batch of timesteps and return a tensor of (batch,embedd_dim)(batch,320)
def time_step_embedding( time_steps: torch.Tensor, max_period: int = 10000, embedd_length = 320):
  half = embedd_length // 2
  frequencies = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=time_steps.device)
  args = time_steps[:, None].float() * frequencies[None]
  return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class SKIP:
  def __init__(self):
    self.l=[]
  def add(self,x):
    return self.l.append(x)
  def last(self):
    return self.l[-1]
  def re(self):
    return self.l.pop()
  def show(self):
    return self.l
skip =SKIP()

class RESNETBLOCK2D(nn.Module):
  def __init__(self,in_channels,out_channels,embedd_length):
    super().__init__()
    self.res1 = nn.Sequential(
      nn.GroupNorm(32,in_channels),
      nn.SiLU(),
      nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
    )
    self.res2 = nn.Sequential(
      nn.GroupNorm(32,out_channels),
      nn.SiLU(),
      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
    )
    self.time_embeds_expand = nn.Sequential(
      nn.Linear(embedd_length,out_channels),
      nn.SiLU()
    )
    if in_channels==out_channels:
      self.residual_layer = nn.Identity()
    else:
      self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1)
  def forward(self,x,time_embeds):
    # receives time embedding (batch,1280) and tensor from previous layer
    x = x
    x1 = self.res1(x)
    time_embeds = self.time_embeds_expand(time_embeds)
    x1 = x1 + time_embeds.unsqueeze(2).unsqueeze(3)
    x2 = self.res2(x1)
    return x2 + self.residual_layer(x)


class BasicTransformerBlock(nn.Module):
  def __init__(self,in_channels,num_heads,prompt_embed_length):
    super().__init__()
    self.attention = nn.MultiheadAttention(embed_dim=in_channels,num_heads=num_heads,batch_first=True)
    self.cross_attention = nn.MultiheadAttention(embed_dim=in_channels,kdim=prompt_embed_length,\
      vdim=prompt_embed_length,num_heads=num_heads,batch_first=True,dropout=0.4)
    self.feed_forward = nn.Sequential(
      nn.Linear(in_channels,in_channels*4),
      nn.GELU(),
      nn.Dropout(0.3),
      nn.Linear(in_channels*4,in_channels),
      nn.Dropout(0.3)
    )
    self.layer_norm1 = nn.LayerNorm(in_channels)
    self.layer_norm2 = nn.LayerNorm(in_channels)
    self.layer_norm3 = nn.LayerNorm(in_channels)
    self.layer_norm4 = nn.LayerNorm(prompt_embed_length)
    
  def forward(self,x,prompt):
    x = x
    x1 = self.layer_norm1(x)
    x1,_ = self.attention(x1,x1,x1)
    x = x1 + x
    x2 = self.layer_norm2(x)
    prompt = self.layer_norm4(prompt)
    x2,_ = self.cross_attention(x2,prompt,prompt)
    x = x2 + x
    x3 = self.layer_norm3(x)
    x3 = self.feed_forward(x3)
    x = x3 + x
    return x


class TRANSFORMERBLOCK2D(nn.Module):
  def __init__(self,in_channels,num_heads,prompt_embed_length):
    super().__init__()
    self.g_n = nn.GroupNorm(32,in_channels,eps=1e-6)
    self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1)
    self.transfomerblock = BasicTransformerBlock(in_channels,num_heads,prompt_embed_length)
    self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1)

  def forward(self,x,prompt):
    h,w = x.shape[2],x.shape[3]
    x = x
    x1 = self.g_n(x)
    x1 = self.conv1(x1)
    x1 = torch.permute(x1,(0,2,3,1))
    x1 = torch.reshape(x1,(x1.shape[0],h*w,-1))
    x1 = self.transfomerblock(x1,prompt)
    x1 = torch.reshape(x1,(x1.shape[0],h,w,-1))
    x1 = torch.permute(x1,(0,3,1,2))
    x1 = self.conv2(x1)
    return x1 + x

class DOWNSAMPLE2D(nn.Module):
  def __init__(self,in_channels):
    super().__init__()
    self.down = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=2,padding=1)
  def forward(self,x):
    return self.down(x)

class CROSSATTNDOWNBLOCK2D(nn.Module):
  def __init__(self,in_channels,out_channels,num_heads,prompt_embed_length,embedd_length):
    super().__init__()
    self.module_list_cross = nn.ModuleList(modules=(RESNETBLOCK2D(in_channels,out_channels,embedd_length),
      TRANSFORMERBLOCK2D(out_channels,num_heads,prompt_embed_length),
      RESNETBLOCK2D(out_channels,out_channels,embedd_length),
      TRANSFORMERBLOCK2D(out_channels,num_heads,prompt_embed_length),
      DOWNSAMPLE2D(out_channels)))
  def forward(self,x,time_embeds,prompt):
    for layer in self.module_list_cross:
      if isinstance(layer,RESNETBLOCK2D):
        x = layer(x,time_embeds)
      elif isinstance(layer,TRANSFORMERBLOCK2D):
        x = layer(x,prompt)
        skip.add(x)
      else:
        x = layer(x)
        skip.add(x)
    return x

class DOWNBLOCK2D(nn.Module):
  def __init__(self,in_channels,out_channels,embedd_length):
    super().__init__()
    self.module_list_down = nn.ModuleList(modules=(RESNETBLOCK2D(in_channels,out_channels,embedd_length),
    RESNETBLOCK2D(out_channels,out_channels,embedd_length)))
  def forward(self,x,time_embeds):
    for layer in self.module_list_down:
      x = layer(x,time_embeds)
      skip.add(x)
    return x

class UNETMIDBLOCK2DCROSSATT(nn.Module):
  def __init__(self,in_channels,out_channels,num_heads,prompt_embed_length,embedd_length):
    super().__init__()
    self.resmid1 = RESNETBLOCK2D(out_channels,out_channels,embedd_length)
    self.resmid2 = RESNETBLOCK2D(out_channels,out_channels,embedd_length)
    self.transmid = self.trans2 = TRANSFORMERBLOCK2D(out_channels,num_heads,prompt_embed_length)
  def forward(self,x,time_embeds,prompt):
    x = self.resmid1(x,time_embeds)
    x = self.transmid(x,prompt)
    x = self.resmid2(x,time_embeds)
    return x

class UPSAMPLE2D(nn.Module):
  def __init__(self,in_channels,out_channels):
    super().__init__()
    self.convup = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
    self.up = nn.Upsample(scale_factor=2,mode='nearest')
  def forward(self,x):
    x = self.up(x)
    x = self.convup(x)
    return x


class UPBLOCK2D(nn.Module):
  def __init__(self,in_channels,out_channels,embedd_length):
    super().__init__()
    self.modules_list_up = nn.ModuleList(modules=(RESNETBLOCK2D(in_channels*2,out_channels,embedd_length),
    RESNETBLOCK2D(out_channels*2,out_channels,embedd_length),
     RESNETBLOCK2D(out_channels*2,out_channels,embedd_length),
     UPSAMPLE2D(out_channels,out_channels)))
  def forward(self,x,time_embeds):
    for layer in self.modules_list_up:
      if isinstance(layer,RESNETBLOCK2D):
        x = torch.cat((x,skip.re()),dim=1)
        x = layer(x,time_embeds)
      else:
        x = layer(x)
    return x


class CROSSATTNUPBLOCK2D(nn.Module):
  def __init__(self,in_channels,out_channels,num_heads,embedd_length,prompt_embed_length):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.embedd_length = embedd_length
    self.module_list_crossup = nn.ModuleList(modules=(
      RESNETBLOCK2D(512,out_channels,embedd_length),
      TRANSFORMERBLOCK2D(out_channels,num_heads,prompt_embed_length),
      RESNETBLOCK2D(384,out_channels,embedd_length),
      TRANSFORMERBLOCK2D(out_channels,num_heads,prompt_embed_length),
      RESNETBLOCK2D(384,out_channels,embedd_length),
      TRANSFORMERBLOCK2D(out_channels,num_heads,prompt_embed_length),
      UPSAMPLE2D(out_channels,out_channels)
    ))
  def forward(self,x,time_embeds,prompt):
    # device = x.device
    for i,layer in enumerate(self.module_list_crossup):
      if isinstance(layer,RESNETBLOCK2D):
        x = torch.cat((x,skip.re()),dim=1)
        x = layer(x,time_embeds)
        # if x.shape[1]!=(self.in_channels):
        #   self.module_list_crossup[i] = RESNETBLOCK2D(x.shape[1],self.out_channels,self.embedd_length).to(device)
        #   x = self.module_list_crossup[i](x,time_embeds)
        # else:
        #   x = self.module_list_crossup[i](x,time_embeds)
      elif isinstance(layer,TRANSFORMERBLOCK2D):
        x = layer(x,prompt)
      else:
        x = layer(x)
    # self.in_channels = x.shape[1]
    return x
class CROSSATTNUPBLOCK2DSPECIAL(nn.Module):
  def __init__(self,in_channels,out_channels,num_heads,embedd_length,prompt_embed_length):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.embedd_length = embedd_length
    self.module_list_crossup = nn.ModuleList(modules=(
      RESNETBLOCK2D(384,out_channels,embedd_length),
      TRANSFORMERBLOCK2D(out_channels,num_heads,prompt_embed_length),
      RESNETBLOCK2D(384,out_channels,embedd_length),
      TRANSFORMERBLOCK2D(out_channels,num_heads,prompt_embed_length),
      RESNETBLOCK2D(256,out_channels,embedd_length),
      TRANSFORMERBLOCK2D(out_channels,num_heads,prompt_embed_length),
    ))
  def forward(self,x,time_embeds,prompt):
    device = x.device
    for i,layer in enumerate(self.module_list_crossup):
      if isinstance(layer,RESNETBLOCK2D):
        x = torch.cat((x,skip.re()),dim=1)
        x = layer(x,time_embeds)
        # if x.shape[1]!=(self.in_channels):
        #   self.module_list_crossup[i] = RESNETBLOCK2D(x.shape[1],self.out_channels,self.embedd_length).to(device)
        #   x = self.module_list_crossup[i](x,time_embeds)
        # else:
        #   x = self.module_list_crossup[i](x,time_embeds)
      elif isinstance(layer,TRANSFORMERBLOCK2D):
        x = layer(x,prompt)
    # self.in_channels = x.shape[1]
    return x

class TIME_EMBEDDS_EXPAND(nn.Module):
  def __init__(self,time_embedds_length_in):
    super().__init__()
    self.tim = nn.Sequential(
      nn.Linear(time_embedds_length_in,time_embedds_length_in*4),
      nn.SiLU(),
      nn.Linear(time_embedds_length_in*4,time_embedds_length_in*4)
    )
  def forward(self,x):
    return self.tim(x)
    
class UNET(nn.Module):
  def __init__(
    self,in_channels,
    out_channels,embedd_length,
    prompt_embed_length,
    num_heads,
    down_block_types:tuple = ("CROSSATTNDOWNBLOCK2D",\
    "CROSSATTNDOWNBLOCK2D","DOWNBLOCK2D"),
    down_block_channels:tuple =(64,128),
    up_block_types:tuple=("UPBLOCK2D","CROSSATTNUPBLOCK2D","CROSSATTNUPBLOCK2DSPECIAL")):
    super().__init__()
    self.encoder = nn.ModuleList()
    self.decoder = nn.ModuleList()
    self.time_expand = TIME_EMBEDDS_EXPAND(320)
    self.cross_blocks = len(down_block_types) -1
    if not (len(down_block_channels)==len(down_block_types)-1):
        assert "down block channels are less in number"

    for i, block in enumerate(down_block_types):
      if block=="CROSSATTNDOWNBLOCK2D":
        if i!=len(down_block_types)-2:
          self.encoder.append(CROSSATTNDOWNBLOCK2D(down_block_channels[i],down_block_channels[i+1],num_heads,prompt_embed_length,embedd_length))
        else:
          self.encoder.append(CROSSATTNDOWNBLOCK2D(down_block_channels[i],down_block_channels[-1],num_heads,prompt_embed_length,embedd_length))

      elif block=="DOWNBLOCK2D":
        self.encoder.append(DOWNBLOCK2D(down_block_channels[-1],down_block_channels[-1],embedd_length))

    for i,block in enumerate(up_block_types):
      if block == "UPBLOCK2D":
        self.decoder.append(UPBLOCK2D(down_block_channels[-1],down_block_channels[-1],embedd_length))
      elif block=="CROSSATTNUPBLOCK2D":
        if i!=self.cross_blocks:
          self.decoder.append(CROSSATTNUPBLOCK2D(down_block_channels[-1],down_block_channels[-1+i],num_heads,embedd_length,prompt_embed_length))
      else:
        self.decoder.append(CROSSATTNUPBLOCK2DSPECIAL(down_block_channels[0],down_block_channels[0],num_heads,embedd_length,prompt_embed_length))
    
    self.mid = UNETMIDBLOCK2DCROSSATT(down_block_channels[-1],down_block_channels[-1],num_heads,prompt_embed_length,embedd_length)
    self.conv_initial = nn.Conv2d(in_channels,down_block_channels[0],kernel_size=3,stride=1,padding=1)
    self.conv_final = nn.Conv2d(down_block_channels[0],out_channels,kernel_size=3,stride=1,padding=1)
    self.group_norm = nn.GroupNorm(32,down_block_channels[0])
    self.act = nn.SiLU()
  def forward(self,x,time_embeds,prompt):
    time_embeds = time_step_embedding(time_embeds)
    time_embeds = self.time_expand(time_embeds)
    x = self.conv_initial(x)
    skip.add(x)
    for layer in self.encoder:
      if isinstance(layer,CROSSATTNDOWNBLOCK2D):
        x = layer(x,time_embeds,prompt)
      elif isinstance(layer,DOWNBLOCK2D):
        x = layer(x,time_embeds)
    x = self.mid(x,time_embeds,prompt)
    for layer in self.decoder:
      if isinstance(layer,UPBLOCK2D):
        x = layer(x,time_embeds)
      elif isinstance(layer,CROSSATTNUPBLOCK2D):
        x = layer(x,time_embeds,prompt)
      else:
        x = layer(x,time_embeds,prompt)
    x = self.group_norm(x)
    x = self.act(x)
    x = self.conv_final(x)
    return x



# x = torch.randn(10,4,32,32)
# y = torch.tensor([4,6,7,7,65,54,44,3,2,1])
# print(y.shape)
# z = torch.randn(10,77,312)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# unet = UNET(in_channels=4,out_channels=4,embedd_length=1280,
# prompt_embed_length=312,down_block_types=("CROSSATTNDOWNBLOCK2D",
#     "CROSSATTNDOWNBLOCK2D","DOWNBLOCK2D"),down_block_channels=(128,256),
#     up_block_types=("UPBLOCK2D","CROSSATTNUPBLOCK2D","CROSSATTNUPBLOCK2DSPECIAL"),num_heads=8).to(device)
# # unet(x,y,z)
# from torchsummary import summary
# from celeb_data import dataloader
# # summary(unet)
# from vae_pre import VAE
# from text_encoder import BERT
# from diffusion import diffusion
# loss_fn = nn.MSELoss(reduction='none')
# bert_encoder = BERT(device)
# diff = diffusion(device)
# vae = VAE().to(device)
# images,texts = next(iter(dataloader))
# images = images.to(device)
# text_encodings = bert_encoder(texts)
# latent = vae.latent(images)
# noise_images,noise,timesteps = diff.noised_images(latent)
# timesteps = timesteps.to(device)
# pred_noise = unet(noise_images,timesteps.squeeze(1),text_encodings)
# loss  = loss_fn(pred_noise,noise)
# loss = loss.reshape(loss.shape[0],-1).sum(dim=1)
# loss = loss.mean()
# print(loss)
# print(skip.show())





    




