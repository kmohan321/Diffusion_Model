import torch
import torch.nn as nn
import math
from torchsummary import summary

class SelfAttention(nn.Module):
  def __init__(self,in_size):
    super().__init__()
    self.norm = nn.LayerNorm(in_size)
    self.MHA = nn.MultiheadAttention(embed_dim=in_size,num_heads=4)
    self.seq2 = nn.Sequential(
      nn.LayerNorm(in_size),
      nn.Linear(in_size,in_size),
      nn.GELU(),
      nn.Linear(in_size,in_size)
    )
  def forward(self,x):
    x = torch.reshape(x,(x.shape[0],x.shape[1],-1))
    x = torch.swapaxes(x,2,1)
    x = self.norm(x)
    y,_ = self.MHA(x,x,x)
    x = x + y
    x = y + self.seq2(x)
    x = torch.swapaxes(x,1,2)
    return x.view(x.shape[0],x.shape[1],int(math.sqrt(x.shape[2])),int(math.sqrt(x.shape[2])))

class VAE(nn.Module):
  def __init__(self, latent_dim,device):
        super(VAE, self).__init__()
        self.device = device
        
        # Encoder
        self.encoder = nn.Sequential(
            self._conv_block(3, 64),  
            self._conv_block_repeat(64,64), 
            SelfAttention(64),  
            self._conv_block_repeat(64, 128),
            self._conv_block_repeat(128,128),  
            SelfAttention(128) ,
            self._conv_block(128,256),
            self._conv_block_repeat(256,256),
            SelfAttention(256),
            self._conv_block_repeat(256,512),
            self._conv_block_repeat(512,512),
            nn.Conv2d(512,8,kernel_size=3,stride=1,padding=1)  
        )
        
        # Latent space
        self.mean = nn.Conv2d(8,latent_dim,kernel_size=3,padding=1,stride=1)
        self.log_var = nn.Conv2d(8,latent_dim,kernel_size=3,padding=1,stride=1)
        
        # Decoder
        self.decoder_input = nn.Conv2d(latent_dim,512,kernel_size=3,stride=1,padding=1)
        self.decoder = nn.Sequential(
            self._conv_transpose_block(512,256),
            self._conv_transpose_block_repeat(256, 256), 
            SelfAttention(256),
            self._conv_transpose_block_repeat(256, 128), 
            self._conv_transpose_block_repeat(128, 128), 
            SelfAttention(128), 
            self._conv_transpose_block_repeat(128, 64),
            self._conv_transpose_block_repeat(64, 64),
            SelfAttention(64),   
            self._conv_transpose_block(64, 3, last_layer=True)  
        )

  def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
  def _conv_block_repeat(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
  def _conv_transpose_block_repeat(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
  def _conv_transpose_block(self, in_channels, out_channels, last_layer=False):
    if last_layer:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        nn.LeakyReLU()
    )

  def reparameterization(self,mu,log_var):
    noise = torch.randn_like(mu).to(self.device)
    z = mu + noise * torch.exp(log_var/2)
    return z


  def forward(self,x):
    x = self.encoder(x)
    self.mu,self.log_variance = self.mean(x),self.log_var(x)
    z = self.reparameterization(self.mu,self.log_variance)
    return z,self.mu,self.log_variance
  
  def forward_decoder(self,x):
    # z = self.forward(x)
    # print(z.shape)
    z = self.decoder_input(x)
    return self.decoder(z)
    
  def inference_model(self,x):
    self.mu,self.log_variance = self.mean(x),self.log_var(x)
    z = self.mu + torch.exp(self.log_variance/2)
    return z


# x = torch.randn((16,3,128,128)).to('cuda')
# vae = VAE(3,'cuda').to('cuda')
# out= vae.forward(x)
# out,_,_ = vae.forward_decoder(out)
# print(out.shape)
# summary(vae)

