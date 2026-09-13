import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup_model_for_finetuning(num_classes):
    """
    Sets up the ResNet-50 model architecture.
    app.py imports this function to build the AI's 'skeleton' before loading weights.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def main():
    # 1. Setup GPU Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Training on: {device}")

    # 2. Data Augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load the Dataset
    data_dir = r"E:\SAT-QUERY AI\satquery_ai\satquery_ai\images\EuroSAT_RGB"
    
    print(f"📂 Loading images from: {data_dir}")
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # 4. Split into Training (80%) and Validation (20%) Automatically
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    num_classes = len(full_dataset.classes)
    
    print("\n" + "="*50)
    print(f"⚠️ IMPORTANT: You now have {num_classes} classes.")
    print(f"Copy this exact array into your app.py later:")
    print(f"{full_dataset.classes}")
    print("="*50 + "\n")

    # 5. Rebuild the ResNet-50 Brain using the helper function
    model = setup_model_for_finetuning(num_classes)
    model = model.to(device)

    # 6. Math & Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7. Train the AI
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Batch [{i}/{len(train_loader)}] - Loss: {loss.item():.4f}")
            
        print(f"✅ Epoch {epoch+1} Completed. Average Loss: {running_loss/len(train_loader):.4f}\n")

    # 8. Save the New Brain
    save_path = 'satquery_custom_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"🎉 Training Complete! Model saved as '{save_path}'")

if __name__ == '__main__':
    main()