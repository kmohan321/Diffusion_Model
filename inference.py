from text_encoder import BERT
from unetnew import UNET
from diffusion import diffusion
import torch
import random
from vae_pre import VAE
from torchvision.utils import save_image
device = 'cuda' if torch.cuda.is_available() else 'cpu'

file_path = 'saved_images\images_test.png'
load_model_unet = True

unet = UNET(in_channels=4,out_channels=4,embedd_length=1280,
prompt_embed_length=312,down_block_types=("CROSSATTNDOWNBLOCK2D",
    "CROSSATTNDOWNBLOCK2D","DOWNBLOCK2D"),down_block_channels=(128,256),
    up_block_types=("UPBLOCK2D","CROSSATTNUPBLOCK2D","CROSSATTNUPBLOCK2DSPECIAL"),num_heads=8).to(device)
bert_encoder = BERT(device)
vae = VAE().to(device)
diff = diffusion(device)

if load_model_unet==True:
  unet.load_state_dict(torch.load('unet_model.pth'))

attribute_names = [
            "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
            "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
            "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
            "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
            "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
            "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
            "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
            "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
        ]
l =[]
num_images=16
for i in range(num_images):
  random_attr = random.choice(attribute_names)
  l.append(random_attr)

text_encodings = bert_encoder(l)
sampled_img = diff.sampling(unet,text_encodings)
with torch.no_grad():
      sampled_img = vae.decoder(sampled_img)
save_image(sampled_img,file_path)
print(l)