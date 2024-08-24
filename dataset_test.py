from datasets import load_dataset
from PIL import Image
import io

base_url = "https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-high-quality-captions/resolve/main/data/data-{i:06d}.tar"
num_shards = 66  # Number of webdataset tar files
urls = [base_url.format(i=i) for i in range(num_shards)]

dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)

# Iterate through the dataset
for item in dataset:
    # The image data is typically stored in the 'jpg' key
    if 'jpg' in item:
        image = item['jpg']
        
        # Check if image is already a PIL Image object
        if isinstance(image, Image.Image):
            # Save the image
            image.save("output_image.jpg")
            print("Image saved as output_image.jpg")
        else:
            print("Unexpected image format")
    else:
        print("No image data found in this item")
    
    # Break after processing one image
    break

# Print the entire item for debugging
print("Item contents:")
for key, value in item.items():
    print(f"{key}: {type(value)}")