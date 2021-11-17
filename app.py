import streamlit as st
from net import ColorizationNet

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
from skimage.transform import resize
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision import datasets, transforms

import os, shutil, time, base64

model = ColorizationNet()
pretrained = torch.load('model-epoch-109-losses-16962.837.pth', map_location=lambda storage, loc: storage)
model.load_state_dict(pretrained)
model = model.cpu()

colorize_image = None
save_button = None

video_in = './video_outputs/'

def preprocess_input(img):
    gray = io.imread(img)
    gray = rgb2gray(gray)

    gray = resize(gray, (224, 224))

    gray_input = np.asarray(gray)
    gray_input = torch.from_numpy(gray_input).unsqueeze(0).float()
    gray_input = gray_input.unsqueeze(1)
    
    return gray_input


def postprocess_output(gray_input, output):
    plt.clf()

    # combine channels
    color_image = torch.cat((gray_input[0], output[0]), 0).detach().numpy() 
    # rescale for matplotlib
    color_image = color_image.transpose((1, 2, 0))

    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   

    color_image = lab2rgb(color_image.astype(np.float64))

    return color_image


def infer(file):
    model.eval()

    gray_input = preprocess_input(file)
    output = model(gray_input)
    color_image = postprocess_output(gray_input, output)
    
    input_image = Image.open(file)
    st.image(input_image, caption = 'Input B/W Image', use_column_width=True)
    st.image(color_image, caption = 'Output colorized image', use_column_width=True)

    return color_image

st.markdown("<h1 style='text-align: center;'>Neural Network Colorizer</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Using the LAB color space to color images</h5>", unsafe_allow_html=True)

image_file = st.file_uploader("Upload B/W Image (JPG / JPEG)", type = ["jpg","jpeg"])
colorize_image = st.button("Colorize Image")

if colorize_image:
    if image_file is None:
        st.write("Please upload an image")
    else:
        with st.spinner('Model working....'):
            color_image = infer(image_file)
            plt.imsave(arr = color_image, fname = 'output.jpeg')
            
            with open('output.jpeg', 'rb') as file:
                download_button = st.download_button (label = "Download Image", data = file, file_name = "output.jpeg", 
                mime = "image/jpeg")