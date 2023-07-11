import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
from model import InpaintingGAN
from dataset import get_dataloader


def train_model(model, dataloader, num_epochs, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for i, images in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.size(0)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, images)

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Lưu trọng số mô hình sau mỗi epoch
        checkpoint_path = os.path.join("inpainting/models", f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    # Cài đặt thông số
    root_dir = "dataset/original"
    batch_size = 16
    num_workers = 4
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tạo DataLoader
    dataloader = get_dataloader(root_dir, batch_size, num_workers)

    # Khởi tạo mô hình inpainting
    model = InpaintingGAN()

    # Huấn luyện mô hình
    train_model(model, dataloader, num_epochs, device)
