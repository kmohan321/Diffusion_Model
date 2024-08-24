from diffusers import UNet2DConditionModel
from torchsummary import summary
model = UNet2DConditionModel(sample_size=128,in_channels=4,out_channels=4)
# summary(model)