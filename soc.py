import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models

# Constants
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 4
NUM_WORKERS = 2
PIN_MEMORY = DEVICE == "cuda"
TRAIN_IMG_DIR = "C:/train/image"
TRAIN_MASK_DIR = "C:/train/mask"
VAL_IMG_DIR = "C:/Users/marcospulin7/Downloads/test/image"
VAL_MASK_DIR = "C:/Users/marcospulin7/Downloads/test/mask"

class PersonSegmentData(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask

def get_data_loaders(train_dir, train_mask_dir, val_dir, val_maskdir, batch_size,
                     train_transform, val_transform, num_workers=4, pin_memory=True):
    train_ds = PersonSegmentData(train_dir, train_mask_dir, transform=train_transform)
    val_ds = PersonSegmentData(val_dir, val_maskdir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, shuffle=False)
    return train_loader, val_loader

def save_checkpoint(state, filename="resize.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# Transforms
train_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])
val_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

class ResNet_50(nn.Module):
    def __init__(self, output_layer=None):
        super(ResNet_50, self).__init__()
        self.pretrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers) - self.layer_count):
            self.pretrained._modules.pop(self.layers[-i])
        self.net = nn.Sequential(*list(self.pretrained.children()))

    def forward(self, x):
        return self.net(x)

class Atrous_Convolution(nn.Module):
    def __init__(self, input_channels, kernel_size, pad, dilation_rate, output_channels=256):
        super(Atrous_Convolution, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size,
                              padding=pad, dilation=dilation_rate, bias=False)
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class ASSP(nn.Module):
    def __init__(self, in_channles, out_channles):
        super(ASSP, self).__init__()
        self.conv_1x1 = Atrous_Convolution(in_channles, 1, 0, 1, out_channles)
        self.conv_6x6 = Atrous_Convolution(in_channles, 3, 6, 6, out_channles)
        self.conv_12x12 = Atrous_Convolution(in_channles, 3, 12, 12, out_channles)
        self.conv_18x18 = Atrous_Convolution(in_channles, 3, 18, 18, out_channles)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channles, out_channles, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True)
        )
        self.final_conv = Atrous_Convolution(out_channles * 5, 1, 0, 1, out_channles)

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_6x6 = self.conv_6x6(x)
        x_12x12 = self.conv_12x12(x)
        x_18x18 = self.conv_18x18(x)
        img_pool = self.image_pool(x)
        img_pool = F.interpolate(img_pool, size=x_18x18.shape[2:], mode='bilinear', align_corners=True)
        concat = torch.cat([x_1x1, x_6x6, x_12x12, x_18x18, img_pool], dim=1)
        return self.final_conv(concat)

class Deeplabv3Plus(nn.Module):
    def __init__(self, num_classes):
        super(Deeplabv3Plus, self).__init__()
        self.backbone = ResNet_50(output_layer='layer3')
        self.low_level_features = ResNet_50(output_layer='layer1')
        self.assp = ASSP(1024, 256)
        self.conv1x1 = Atrous_Convolution(256, 1, 0, 1, 48)
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        x_backbone = self.backbone(x)
        x_low = self.low_level_features(x)
        x_assp = self.assp(x_backbone)
        x_assp_up = F.interpolate(x_assp, scale_factor=4, mode='bilinear', align_corners=True)
        x_low = self.conv1x1(x_low)
        x_cat = torch.cat([x_low, x_assp_up], dim=1)
        x_final = self.conv_3x3(x_cat)
        x_final = F.interpolate(x_final, scale_factor=4, mode='bilinear', align_corners=True)
        return self.classifier(x_final)

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        bce = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = 1 - (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return bce + dice

class IOU(nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() - intersection
        return (intersection + smooth) / (union + smooth)

def show_transformed(data_loader):
    batch = next(iter(data_loader))
    images, masks = batch

    for i, (img, mask) in enumerate(zip(images, masks)):
        if i == 5: break
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
        plt.title("Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask.cpu().numpy(), cmap='gray')
        plt.title("Mask")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
        BATCH_SIZE, train_transform, val_transforms,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    model = Deeplabv3Plus(num_classes=1).to(DEVICE)
    loss_fn = DiceBCELoss()
    iou_fn = IOU()

    use_cuda = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_iou = []
    train_loss = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}")
        model.train()
        iter_loss, iter_iou = 0.0, 0.0
        iterations = 0
        batch_loop = tqdm(train_loader)

        for data, targets in batch_loop:
            data = data.to(DEVICE)
            targets = targets.float().unsqueeze(1).to(DEVICE)

            with torch.amp.autocast(device_type="cuda"):
                predictions = model(data)
                loss = loss_fn(predictions, targets)
                iou = iou_fn(predictions, targets)

            iter_loss += loss.item()
            iter_iou += iou.item()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iterations += 1
            batch_loop.set_postfix(diceloss=loss.item(), iou=iou.item())

        train_loss.append(iter_loss / iterations)
        train_iou.append(iter_iou / iterations)
        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}, Training loss: {round(train_loss[-1], 3)}")

        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

        model.eval()
        num_correct = 0
        num_pixels = 0
        dice_score = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum().item()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum().item()) / ((preds + y).sum().item() + 1e-8)
                torch.cuda.empty_cache()

        print(f"Got {num_correct}/{num_pixels} with acc {100 * num_correct / num_pixels:.2f}%")
        print(f"Dice score: {dice_score / len(val_loader)}")

    plt.figure(figsize=(16, 10))
    plt.plot(train_loss, label='Dice BCE Loss')
    plt.plot(train_iou, label='IoU')
    plt.legend()
    plt.show()

    # Show a few transformed validation samples
    show_transformed(val_loader)
