import streamlit as st
import os
from PIL import Image
import tensorflow
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.neighbors import NearestNeighbors
import pickle

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
featurelist = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
st.markdown(
    "<h1 style='text-align: left; color: #262626;'>Fashion App</h1>",
    unsafe_allow_html=True
)
import base64

def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("background.jpg")  # Put your background image in the same folder


def extract_features(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)  # Expand dims to match model input
    preprocessing_img = preprocess_input(expanded_img_array)  # Correct preprocessing
    result = model.predict(preprocessing_img).flatten()
    normalized_result = result / norm(result)  # Normalize feature vector
    return normalized_result
def save(uploaded_file):
    try:
        with open(os.path.join('ups',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0
def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(featurelist)

    distance, indices = neighbors.kneighbors(features)
    return indices
uploaded_fi=st.file_uploader("Upload Image")
if uploaded_fi is not None:
    if save(uploaded_fi):
        display=Image.open(uploaded_fi)
        features=extract_features(os.path.join('ups',uploaded_fi.name),model)
        features = features.reshape(1, -1)
        st.text(features)
        st.image(display)
        indices = recommend(features,featurelist)
        col1,col2,col3,col4,col5=st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured")