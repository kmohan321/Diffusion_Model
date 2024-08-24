import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class diffusion:
  def __init__(self,device,timesteps = 300,beta_low =0.0001, beta_high =0.02):
    self.device = device
    self.timesteps = timesteps
    self.beta_high = beta_high
    self.beta_low = beta_low
    self.get_betas  = torch.linspace(self.beta_low,self.beta_high,steps=self.timesteps).to(self.device)
    self.alphas = 1. - self.get_betas
    self.alpha_hat = torch.cumprod(self.alphas,dim=0)

  def get_noise(self,images):
    return torch.randn((images.shape[0],images.shape[1],images.shape[2],images.shape[3])).to(self.device)
  
  def get_timesteps(self,images):
    return torch.randint(0,self.timesteps-1,(images.shape[0],1))

  def noised_images(self,images):
    noise  = self.get_noise(images)
    timesteps = self.get_timesteps(images)
    req_alpha_hat = self.alpha_hat[timesteps].view(images.shape[0],1,1,1)
    noised_img = torch.sqrt(req_alpha_hat)*images + torch.sqrt(1.-req_alpha_hat)*noise
    return noised_img,noise,timesteps


  def sampling(self,unet,text_encodings,num_images = 16):
    noise_img = torch.randn((num_images,4,16,16)).to(self.device)
    unet.eval()
    for timestep in reversed(range(0,300)):
        timesteps = torch.ones((num_images,1),dtype=torch.long) * timestep
        timesteps = timesteps.to(self.device)
        with torch.no_grad():
          # print(text_encodings[:num_images,:,:].shape)
          pred_noise = unet(noise_img,timesteps.squeeze(1),text_encodings[:num_images,:,:])
        if timestep > 0:
          noise_img = (noise_img - (self.get_betas[timesteps].view(noise_img.shape[0],1,1,1) * pred_noise / (torch.sqrt(1.-self.alpha_hat[timesteps].view(noise_img.shape[0],1,1,1)))) / self.alpha_hat[timesteps].view(noise_img.shape[0],1,1,1)) + torch.sqrt(self.get_betas[timesteps].view(noise_img.shape[0],1,1,1)) * self.get_noise(noise_img)
        else:
          noise_img  = (noise_img - (self.get_betas[timesteps].view(noise_img.shape[0],1,1,1) * pred_noise / (torch.sqrt(1.-self.alpha_hat[timesteps].view(noise_img.shape[0],1,1,1)))) / self.alpha_hat[timesteps].view(noise_img.shape[0],1,1,1)) 
    # sampled_img = ((torch.clamp(noise_img,-1,1).view(noise_img.shape[0],64,64,3) + 1) / 2).type(torch.long)
    return noise_img


# diff  = Diffusion(device)
# diff.sampling()