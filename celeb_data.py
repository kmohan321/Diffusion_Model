import os
import random
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch

class CelebADataset(Dataset):
    def __init__(self, img_dir, annotation_file, transform=None, max_images=None, seed=None):
        self.img_dir = img_dir
        self.transform = transform
        
        self.attribute_names = [
            "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
            "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
            "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
            "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
            "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
            "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
            "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
            "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
        ]
        
        # Read annotations
        all_annotations = self.read_annotations(annotation_file, max_images)
        
        # If max_images is set, randomly select subset
        if max_images and max_images < len(all_annotations):
            if seed is not None:
                random.seed(seed)
            selected_files = random.sample(list(all_annotations.keys()), max_images)
            self.annotations = {k: all_annotations[k] for k in selected_files}
        else:
            self.annotations = all_annotations
        
        self.filenames = list(self.annotations.keys())

    def read_annotations(self, file_path, max_images=None):
        annotations = {}
        with open(file_path, 'r') as f:
            for line in f:
                if max_images and len(annotations) >= max_images:
                    break
                parts = line.strip().split()
                filename = parts[0]
                attributes = list(map(int, parts[1:]))
                annotations[filename] = attributes
        return annotations

    def attributes_to_text(self, attributes):
        present_attributes = [name for attr, name in zip(attributes, self.attribute_names) if attr == 1]
        return " ".join(present_attributes) if present_attributes else "No specific attributes"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        attributes = self.annotations[img_name]
        text_desc = self.attributes_to_text(attributes)
        
        return image, text_desc

# Usage example
from torchvision import transforms
from torchvision.transforms import v2

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    v2.ToDtype(torch.float16)
])


dataset = CelebADataset('D:\img_align_celeba', 'list_attr_celeba.txt', 
                        transform=transform, max_images=19968, seed=42)
dataloader = DataLoader(dataset,batch_size=48,pin_memory=True,shuffle=True)

# images,text = next(iter(dataloader))
# print(images.min(),images.max())
# print(images[0])
# save_image(images[0],'saved_images2\images.png')

# print(len(dataloader))