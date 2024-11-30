import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from transformers import ViTModel
import numpy as np
# Load and preprocess the images (style and content)
def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Transformations for ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT expects fixed-size inputs
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

content_image = load_image("content_temple.jpeg", transform)
style_image = load_image("starryNight.jpg", transform)

# Load the pretrained ViT
vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit.eval()  # Set to evaluation mode

class StyleTransferNetwork(nn.Module):
    def __init__(self, vit):
        super(StyleTransferNetwork, self).__init__()
        self.vit = vit  # Pretrained ViT as the feature extractor
        
        # Decoder architecture with upsampling
        self.decoder = nn.Sequential(
            # First, reduce the feature dimensions
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Upsample to 28x28
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Upsample by 2
            nn.ReLU(),
            
            # Upsample to 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample by 2
            nn.ReLU(),
            
            # Upsample to 112x112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # Upsample by 2
            nn.ReLU(),
            
            # Final layer to upsample to 224x224
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),     # Upsample by 2
            nn.Tanh()  # Output normalized to [-1, 1]
        )

    def forward(self, content_image, style_image):
        # Extract features
        content_features = self.vit(content_image).last_hidden_state
        style_features = self.vit(style_image).last_hidden_state

        # Calculate number of patches (assuming a square grid of patches)
        content_features = content_features[:, 1:, :]  # Remove [CLS] token (1st patch)
        style_features = style_features[:, 1:, :]  # Remove [CLS] token (1st patch)
        combined_features = 0.7 * content_features + 0.3 * style_features
        # Reshape combined features to (batch_size, embedding_dim, height, width)
        # combined_features = combined_features[:, 1:, :] 
        print(f"Content features shape: {content_features.shape}")
        print(f"Style features shape: {style_features.shape}")
        print(f"Combined features shape: {combined_features.shape}")

        combined_features = combined_features.view(1, 768, 14, 14)

        # Decode into image
        output = self.decoder(combined_features)
        return output

    
def compute_content_loss(content_features, target_features):
    return torch.mean((content_features - target_features) ** 2)

def compute_style_loss(style_features, target_features):
    # Gram matrix captures style
    gram_style = torch.matmul(style_features, style_features.transpose(1, 2))
    gram_target = torch.matmul(target_features, target_features.transpose(1, 2))
    return torch.mean((gram_style - gram_target) ** 2)

# Initialize network and optimizer
model = StyleTransferNetwork(vit)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training loop
epochs = 200
for epoch in range(epochs):
    print(f"Epoch [{epoch+1}/{epochs}]")
    optimizer.zero_grad()
    
    # Forward pass
    output = model(content_image, style_image)
    
    # Compute losses
    content_loss = compute_content_loss(model.vit(content_image).last_hidden_state.detach(),
                                        model.vit(output).last_hidden_state)
    style_loss = compute_style_loss(model.vit(style_image).last_hidden_state.detach(),
                                    model.vit(output).last_hidden_state)
    total_loss = content_loss + 1e-1 * style_loss
    
    # Backpropagation
    total_loss.backward()
    optimizer.step()
    
    print(f"Loss: {total_loss.item():.4f}")
    
    # Save the output image only after the last epoch
    if epoch == epochs - 1:
        # Convert the output tensor to a numpy array for saving
        output_image = output.squeeze(0).detach().cpu().numpy()
        output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format
        
        # Denormalize if needed (e.g., for [-1, 1] -> [0, 1])
        output_image = (output_image + 1) / 2.0  # Denormalize to [0, 1]
        
        # Convert to PIL Image format
        output_image = Image.fromarray((output_image * 255).astype(np.uint8))
        
        # Save the output image
        output_image.save(f'output_epoch_{epoch+1}.png')
