import streamlit as st
st.set_page_config(page_title="Satellite Image Segmentation", layout="centered")
# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === UNet Class ===
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor)
        self.up2 = Up(512, 256 // factor)
        self.up3 = Up(256, 128 // factor)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# === Constants ===
NORMALIZATION_MEANS = [0.4643, 0.3185, 0.3141]
NORMALIZATION_STDS = [0.2171, 0.1561, 0.1496]
N_CLASSES = 25
MODEL_PATH = "semantic_segmentation_si_unet.model"  # path to your .model file

# === Load Model ===
@st.cache_resource
def load_model():
    model = UNet(n_channels=3, n_classes=N_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# === Image Preprocessing ===
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.CenterCrop(3000),
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION_MEANS, std=NORMALIZATION_STDS)
    ])
    return transform(image).unsqueeze(0)  # (1, 3, H, W)

# === Convert Mask to Color Image ===
def decode_segmentation_mask(mask, n_classes=N_CLASSES):
    import matplotlib
    colormap = matplotlib.cm.get_cmap('tab20b', n_classes)
    color_mask = colormap(mask / n_classes)[..., :3]  # (H, W, 3)
    return (color_mask * 255).astype(np.uint8)

# === Streamlit UI ===
st.title("🛰️ Satellite Image Segmentation (U-Net)")
st.write("Upload a satellite image, and the model will segment it into 25 classes.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    with st.spinner("Running segmentation..."):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            preds = model(input_tensor)
            pred_labels = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()
            color_mask = decode_segmentation_mask(pred_labels)

    st.success("Segmentation completed!")
    st.image(color_mask, caption="Segmented Mask", use_container_width=True)
