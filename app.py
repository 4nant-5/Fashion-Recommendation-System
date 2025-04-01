import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
import os
# Create the base ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Fix: Instantiate GlobalMaxPooling2D before adding it
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()  # Correct instantiation
])

def extract_features(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = preprocess_input(img_array)
    preprocessing_img=preprocess_input(expanded_img_array)
    result=model.predict(preprocessing_img).flatten()
    normalised_result=result / norm(result)
    return normalised_result

print(os.listdir('images'))