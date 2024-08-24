import torch
import torch.nn as nn
import math
from torchsummary import summary
class ResNet(nn.Module):
  def __init__(self,in_size,out_size):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_size,out_size,kernel_size=3,padding=1),
      nn.GroupNorm(1,out_size),
      nn.GELU(),
      nn.Conv2d(out_size,out_size,kernel_size=3,padding=1),
      nn.GroupNorm(1,out_size)
    )
  def forward(self,x):
    return self.block(x)

class DownSample(nn.Module):
  def __init__(self,in_size,out_size,emb_length):
    super().__init__()
    self.block = nn.Sequential(
      nn.MaxPool2d(kernel_size=2,padding=0),
      ResNet(in_size,out_size),
      ResNet(out_size,out_size)
    )
    self.embed = nn.Sequential(
      nn.SiLU(),
      nn.Linear(emb_length,out_size)
    )
  def forward(self,x,embedding):
    x = self.block(x)
    x = x + self.embed(embedding).view(embedding.shape[0],-1,1,1)
    return x

class Upsample(nn.Module):
  def __init__(self,in_size,out_size,emb_length):
    super().__init__()
    self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

    self.block = nn.Sequential(
      ResNet(in_size*2,out_size), # concating of the attention layers
      ResNet(out_size,out_size)
    )
    self.embed = nn.Sequential(
      nn.SiLU(),
      nn.Linear(emb_length,out_size)
    )
  def forward(self,x,res,embedding):
    x = self.up(x)
    x = torch.cat([x,res],dim=1)
    x = self.block(x)
    x = x + self.embed(embedding).view(embedding.shape[0],-1,1,1)
    return x

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

def time_step_embedding( time_steps: torch.Tensor, max_period: int = 10000, embedd_length = 320):
  half = embedd_length // 2
  frequencies = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=time_steps.device)
  args = time_steps[:, None].float() * frequencies[None]
  return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class CrossAttention(nn.Module):
    def __init__(self, in_size, context_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_size)
        self.context_norm = nn.LayerNorm(context_dim)
        self.MHA = nn.MultiheadAttention(embed_dim=in_size, num_heads=4)
        self.seq = nn.Sequential(
            nn.LayerNorm(in_size),
            nn.Linear(in_size, in_size),
            nn.GELU(),
            nn.Linear(in_size, in_size)
        )
        self.proj = nn.Linear(context_dim, in_size)

    def forward(self, x, context):
        b, c, h, w = x.shape
        # print(x.shape)
        x = x.view(b, c, -1).permute(2, 0, 1)  # (h*w, b, c)
        context = self.proj(self.context_norm(context)).permute(1, 0, 2)  # (6, b, in_size)
        x = self.norm(x)
        y, _ = self.MHA(x, context, context)
        x = x + y
        x = x + self.seq(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)
        return x


class UNET(nn.Module):
    def __init__(self, in_size, out_size, emb_length, context_dim):
        super().__init__()
        self.doubleconv = ResNet(in_size, 64)
        self.down1 = DownSample(64, 128, emb_length)
        self.atten1 = SelfAttention(128)
        self.cross_atten1 = CrossAttention(128, context_dim)
        self.down2 = DownSample(128, 256, emb_length)
        self.atten2 = SelfAttention(256)
        self.cross_atten2 = CrossAttention(256, context_dim)
        self.down3 = DownSample(256, 256, emb_length)
        self.atten3 = SelfAttention(256)
        self.cross_atten3 = CrossAttention(256, context_dim)
        self.bottle = nn.Sequential(
            ResNet(256, 512),
            ResNet(512, 512),
            ResNet(512, 256)
        )
        self.up1 = Upsample(256, 128, emb_length)
        self.atten4 = SelfAttention(128)
        self.cross_atten4 = CrossAttention(128, context_dim)
        self.up2 = Upsample(128, 64, emb_length)
        self.atten5 = SelfAttention(64)
        self.cross_atten5 = CrossAttention(64, context_dim)
        self.up3 = Upsample(64, 64, emb_length)
        self.atten6 = SelfAttention(64)
        self.cross_atten6 = CrossAttention(64, context_dim)
        self.final = nn.Conv2d(64, out_size, kernel_size=1)

    def forward(self, image, embedding, context):
        embedding = embedding.squeeze(-1)
        embedding = time_step_embedding(embedding)
        x1 = self.doubleconv(image)
        x = self.down1(x1, embedding)
        x2 = self.atten1(x)
        x2 = self.cross_atten1(x2, context)
        x = self.down2(x2, embedding)
        x3 = self.atten2(x)
        x3 = self.cross_atten2(x3, context)
        x = self.down3(x3, embedding)
        x = self.atten3(x)
        x = self.cross_atten3(x, context)
        x = self.bottle(x)
        x = self.up1(x, x3, embedding)
        x = self.atten4(x)
        x = self.cross_atten4(x, context)
        x = self.up2(x, x2, embedding)
        x = self.atten5(x)
        x = self.cross_atten5(x, context)
        x = self.up3(x, x1, embedding)
        x = self.atten6(x)
        x = self.cross_atten6(x, context)
        return self.final(x)


# x = torch.randn((64,3,64,64))
# embedding = torch.randn((64,1))
# unet = UNET(4,4,256,512)
# y = unet(x,embedding)
# print(y.shape)
# summary(unet)
print(time_step_embedding(torch.Tensor([23,46,78])).shape)
