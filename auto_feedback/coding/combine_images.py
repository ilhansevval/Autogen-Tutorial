# filename: combine_images.py
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import requests
from io import BytesIO

# Function to download an image from a URL
def download_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

# Function to apply semantic segmentation and extract the person
def extract_person(image, model, device):
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    mask = output_predictions.byte().cpu().numpy()
    return mask

# Function to combine the person with a new background
def combine_images(person_image, background_image, mask):
    # Resize background to match person image
    background_image = background_image.resize(person_image.size)
    # Create a mask for blending
    mask_image = Image.fromarray((mask * 255).astype('uint8'))
    person_cutout = Image.composite(person_image, background_image, mask_image)
    return person_cutout

# URLs of the images
person_image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRCbLonevQIqBWh2Yh2C1BACaaoMhoJIKHedg&usqp=CAU'
background_image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQCm038sxf7zQ09pZ9Jov5dsdoGJNh6YU7uDUza4WUTWimUmFOAIdHnBA1y57wSDVWa4QE&usqp=CAU'

# Download images
person_image = download_image(person_image_url)
background_image = download_image(background_image_url)

# Load the pre-trained DeepLabV3 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet101(pretrained=True).to(device)
model.eval()

# Extract the person from the image
mask = extract_person(person_image, model, device)

# Combine the person with the new background
result_image = combine_images(person_image, background_image, mask)

# Save the resulting image
output_file = 'background.png'
result_image.save(output_file)
print(f"Result saved as {output_file}")