from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

image_path = "hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(image_path)
print(img)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

writer.add_image("Tensor_img", tensor_img)
writer.close()
