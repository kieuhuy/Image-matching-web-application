import streamlit as st
from PIL import Image
@st.cache_data
def home_setup():
    col1,col2,col3 = st.columns([4,1,9])
    with col1:
        image = Image.open('logo.png')
        st.image(image)
    with col2:
        st.write("")
    with col3:
        st.title(" DEMO ")
        st.subheader("Images matching with SuperGlue ")
        st.markdown("  :pick: **Developed by Kieu Chi Huy** :pick: ")
    with st.container():
        st.write("The major objective of the application is to display and visualize my results in a simple and interactive way. ")
        st.write("Please see my report for more information about how the graph is generated and how the machine learning model is handled. ")
        st.write("---")
        st.subheader("Select on the left menu panel what you want to explore:")
        st.markdown(
            """
            - **Information**: This section will give an overview of the Deep Learning model being built and describe its attributes and properties.
            - **Image matching**: This section will provide an Images matching tools in which receive the pair of images and return the matching result.
            """
        )
        st.markdown("**More information can be found by clicking in the READMEs of each tabs.** ")