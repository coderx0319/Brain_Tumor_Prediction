import streamlit as st
from keras_preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the trained model
model = load_model("bestmodel.h5")

def predict_tumor(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return prediction

def main():
    st.title("Brain Tumor Detection")
    st.markdown(
        "Upload an MRI image, and the model will predict if there's a tumor or not."
    )

    uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg"])

    if uploaded_file is not None:
        image_path = "temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.image(uploaded_file, caption="Uploaded MRI Image", width=300)
        st.write("")

        prediction = predict_tumor(image_path)

        if prediction >= 0.5:
            st.error("Prediction: The MRI image has a tumor")
        else:
            st.success("Prediction: The MRI image doesn't have a tumor")

    # Add a link to download the dataset
    st.markdown(
        "If you don't have an MRI image to test, you can download the dataset from [here](https://drive.google.com/drive/folders/17WSYUf4B3Msd17lBpepwZmn_06WWVNkM?usp=drive_link)."
    )

if __name__ == "__main__":
    main()
