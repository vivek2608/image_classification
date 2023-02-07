import streamlit as st
from PIL import Image
from io import StringIO 
import model_utils


st.title('Welcome to Image Classifier')
instructions = """
    This application uses image to classify it as one of the following
    - Buildings
    - Forest
    - Glacier
    - Mountain
    - Sea
    - Street

    Use upload section to upload image. The uploaded images is passed through a deep neural network model 
    in real-time and the out is displayed on the screen.

"""
st.write(instructions)

# upload Image

file = st.file_uploader('Upload An Image')

# Dispaly Image

if file:
    img = Image.open(file)
    st.image(img, width=500, use_column_width=False)
    
    try:
        prediction = model_utils.prediction(img, model_utils.transformer)
        st.title("This image is likely to be "+ prediction)
    except:
        st.error("Image not is desired format!")
