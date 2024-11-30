import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Helper function to load and preprocess the images
def load_image(image_path, size=512, crop=True):
    image = Image.open(image_path)
    if crop:
        # Resize the image while keeping the aspect ratio
        image = transforms.Resize(size)(image)
    # Convert the image to a tensor and normalize it using VGG19's mean and std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to unnormalize and display the image
def imshow(tensor, title=None):
    image = tensor.clone().detach().cpu()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()

# Define AdaIN (Adaptive Instance Normalization) layer
class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content, style):
        # Compute the mean and standard deviation of content and style
        content_mean, content_std = content.mean([2, 3], keepdim=True), content.std([2, 3], keepdim=True)
        style_mean, style_std = style.mean([2, 3], keepdim=True), style.std([2, 3], keepdim=True)
        
        # Perform AdaIN
        normalized_content = (content - content_mean) / content_std
        stylized_content = normalized_content * style_std + style_mean
        return stylized_content

# Load content and style images
content_img = load_image("content_temple.jpeg")
style_img = load_image("starryNight.jpg")

# Pre-trained VGG19 model for feature extraction
vgg = models.vgg19(pretrained=True).features.eval()

# Move the images and model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_img = content_img.to(device)
style_img = style_img.to(device)
vgg = vgg.to(device)

# AdaIN layer
adain = AdaIN().to(device)

# Optimization setup
output_img = content_img.clone().requires_grad_(True)
optimizer = optim.Adam([output_img], lr=0.1)

# Define the loss function
def compute_loss(output, target, style, content):
    # Content loss (MSE)
    content_loss = nn.functional.mse_loss(output, content)
    
    # Style loss (using Gram matrices)
    output_gram = gram_matrix(output)
    style_gram = gram_matrix(style)
    style_loss = nn.functional.mse_loss(output_gram, style_gram)
    
    # Total loss
    total_loss = content_loss + style_loss
    return total_loss

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b, c, h * w)
    tensor_t = tensor.transpose(1, 2)
    gram = torch.bmm(tensor, tensor_t)
    return gram / (c * h * w)

# Style transfer loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Extract features from content and style
    content_features = vgg(content_img)
    style_features = vgg(style_img)
    
    # Transfer style from style image to content image using AdaIN
    output_img = adain(content_features, style_features)

    # Compute loss
    loss = compute_loss(output_img, content_img, style_img, content_img)
    
    # Backpropagation
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Display the output image
imshow(output_img, title='Stylized Image')

# Save the output image
output_image = output_img.squeeze(0).cpu().detach()
output_image = transforms.ToPILImage()(output_image)
output_image.save('stylized_output.png')
