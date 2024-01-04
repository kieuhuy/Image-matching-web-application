import pandas as pd
import streamlit as st
import extra_streamlit_components as stx
import streamlit.components.v1 as components
from PIL import Image
import io
import streamlit as st
import torch
import numpy as np
import argparse
import subprocess
import matplotlib.cm as cm
import os
import shutil
import time
import tempfile
from SuperGluePretrainedNetwork.models import matching
from SuperGluePretrainedNetwork.models.utils import (AverageTimer, VideoStreamer,make_matching_plot_fast, frame2tensor)
#from LoFTR.src.loftr import LoFTR, default_cfg
#from LoFTR.src.utils.plotting import make_matching_figure
torch.set_grad_enabled(False)

def display_uploaded_images():
    st.sidebar.markdown(f"## Welcome to the image matching tool")
    st.sidebar.info(f"Please provide images that you want to match")

    uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True)
    
    images = []
    for index, uploaded_file in enumerate(uploaded_files):
        bytes_data = uploaded_file.read()
        images.append((bytes_data, index + 1))
        
    with st.expander("**Uploaded images**"):
        for image_tuple in images:
            bytes_data, position = image_tuple
            pil_image = Image.open(io.BytesIO(bytes_data))
            st.image(pil_image, caption=f"Picture: {position}", use_column_width=True)
  

    placeholder = st.sidebar.container()        
    option = placeholder.selectbox(f'## Select method:',('SuperGlue','LoFTR'))

    if option == "SuperGlue":
        if st.sidebar.button("Match"):

            if uploaded_files:
                temp_dir = "./temp"
                os.makedirs(temp_dir, exist_ok=True)

                output_dir = "./Result_SuperGlue"
                os.makedirs(output_dir, exist_ok=True)
                # Save the uploaded images to the temporary directory
                for uploaded_file in uploaded_files:
                    img = Image.open(uploaded_file)
                    img.save(os.path.join(temp_dir, uploaded_file.name))

                # Run SuperGlue on the temporary directory
                superglue_command(temp_dir, output_dir)
                img_files = os.listdir(output_dir)
                for img in img_files:
                    image_path = os.path.join(output_dir, img)
                    image = Image.open(image_path)
                    st.image(image, use_column_width=True)

                #Clear old matching result to get the latest data
                if st.button("New match"):
                    shutil.rmtree(output_dir)

                # Delete the temporary directory and its contents
                shutil.rmtree(temp_dir)
            else:
                st.sidebar.warning("Please upload at least two images.")

    if option == "LoFTR":
        if st.sidebar.button("Match"):

            if uploaded_files:
                temp_dir = "./Temp_LoFTR"
                os.makedirs(temp_dir, exist_ok=True)

                output_dir = "./Result_LoFTR"
                os.makedirs(output_dir, exist_ok=True)
            
                for uploaded_file in uploaded_files:
                        img = Image.open(uploaded_file)
                        img_path = os.path.join(temp_dir, uploaded_file.name)
                        img.save(img_path)
                if len(images) == 2:
                    result = loftr_command(temp_dir,output_dir)
                    st.image(result)
                shutil.rmtree(temp_dir)
                if st.button("New match"):
                    shutil.rmtree(output_dir)
            else:
                st.sidebar.warning("Please upload two images")
                

def preprocess_image(images):
    if images is None:
        print("No images found.")
        return []
   
    process_img = []
    for image in images:
        image = cv2.imread(image)
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = gray_image.astype(np.float32) / 255.
            process_img.append(gray_image)
        else: 
            print(f"Error in reading the images:{images}")

def convert_img (images):
    decoded_images = []

    for image_tuple in images:
        bytes_data, _ = image_tuple
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        decoded_images.append(img)

    return decoded_images

def superglue_command(input_dir, output_dir):
       # Progress message
    text = "Operation in progress. Please wait for a moment..."
    progress_bar = st.progress(0)
    for p in range (100):
        time.sleep(0.1)
        progress_bar.progress(p +1, text=text)
    st.success("Superglue model processing completed!",icon="✅")
    progress_bar.empty()
 
    # Construct the command
    command = f"python .\SuperGluePretrainedNetwork\demo_superglue.py --input {input_dir} --output_dir {output_dir} --no_display"

    # Run the command using subprocess
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    # Return the command output and error
    return output.decode("utf-8"), error.decode("utf-8")

def loftr_command(temp_folder,output_dir):
    text = "Operation in progress. Please wait for a moment..."
    progress_bar = st.progress(0)
    for p in range (100):
        time.sleep(0.1)
        progress_bar.progress(p +1, text=text)
    st.success("LoFTR model processing completed!",icon="✅")
    progress_bar.empty()

    model = LoFTR(config=default_cfg)
    model.load_state_dict(torch.load("LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
    model = model.eval()

    image_files = os.listdir(temp_folder)
    if len(image_files) < 2:
        raise ValueError("At least two images are required in the specified folder.")
    
    img0_path = os.path.join(temp_folder, image_files[0])
    img1_path = os.path.join(temp_folder, image_files[1])
    

    img0_raw = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    if img0_raw is not None and img1_raw is not None:
            img0_raw = cv2.resize(img0_raw, (640, 480))
            img1_raw = cv2.resize(img1_raw, (640, 480))
    else:
            # Handle the case where one or both images couldn't be read
            raise ValueError("One or both images couldn't be read.")

    img0 = torch.from_numpy(img0_raw)[None][None] / 255.
    img1 = torch.from_numpy(img1_raw)[None][None] / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        model(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    color = cm.jet(mconf, alpha=0.7)
    text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]
    fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'matching_result.png')
    fig.savefig(output_path, format='png')

    return output_path


  



        
    

