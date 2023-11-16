# filename: semantic_segmentation.py
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import requests
from io import BytesIO

# Function to apply semantic segmentation and save the mask
def semantic_segmentation(image_url, output_file):
    # Download the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    # Define the transformation
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Check if a GPU is available and if not, use a CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained DeepLabV3 model
    model = deeplabv3_resnet101(pretrained=True).to(device)
    model.eval()

    # Perform the segmentation
    with torch.no_grad():
        output = model(input_batch.to(device))['out'][0]
    output_predictions = output.argmax(0)

    # Create a mask with white color for the segmented class
    mask = output_predictions.byte().cpu().numpy()
    mask_image = Image.fromarray(mask * 255)

    # Save the mask
    mask_image.save(output_file)

# URL of the image
image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRCbLonevQIqBWh2Yh2C1BACaaoMhoJIKHedg&usqp=CAU'
# Output file name
output_file = 'mask.png'

# Call the function
semantic_segmentation(image_url, output_file)
print(f"Segmentation mask saved as {output_file}")