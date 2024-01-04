import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from home_setup import *
from image_matching import *
st.set_page_config(
    page_title="Images matching",
    page_icon="👋",
    layout = "wide"
)

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options =["Home", "Instruction","Matching Images"], 
        icons = ['house','book','images'],
        menu_icon = 'list',
        default_index = 0,
    )

if selected == "Home":  
   home_setup()

if selected == "Matching Images":
   st.title("Images matching with SuperGlue")
   with st.expander("**README**"):
      st.markdown(
         '''
         - Provide a folder or picture file containing the images you wish to match in the left menu panel.
         - If the images are successfully uploaded, a container containing them will be present in the main area. Every time an image is added or removed, the container will also be updated with the latest data. 
         - Click the Match button below to run the model 
         - The matching results will appear in the page's main section once the model has finished running. If you would like to match another picture, please remember to click the new match button to make sure you have the latest result.
         '''
      )
   uploade_image = display_uploaded_images()
   preprocess_image(uploade_image)
