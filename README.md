# 🛰️ Satellite Image Segmentation (U-Net)

U-Net–based model for 25-class satellite image segmentation.  
Includes a Streamlit app for uploading an image and viewing the segmented mask.
# Run
streamlit run app.py
# Files
app.py — Streamlit inference app

Segmentation_Satellite_Image.ipynb — Training notebook

semantic_segmentation_si_unet.model — Trained weights

# Requirements
PyTorch, Streamlit, Pillow, Matplotlib, Torchvision.

# Features
Custom U-Net
25-class prediction
Color-coded segmentation output
