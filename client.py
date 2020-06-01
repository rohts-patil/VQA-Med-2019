import streamlit as st
from PIL import Image
import json
import numpy as np
import requests

st.title("VAQ Medical 2019")
question = st.text_input("Question Box", "Which type of organ is shown in scan?")
uploaded_file = st.file_uploader("Please choose an image.", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=200, use_column_width=False)
    st.write("")
    st.markdown("### Preparing the answer to your question...")
    url = "http://127.0.0.1:5000/predict_answer"

    image = json.dumps(np.array(image).tolist())
    response = requests.post(url, json={"question": question, "image": image})
    parsed = json.loads(response.text)
    print(json.dumps(parsed, indent=2))
    st.markdown('### The answer is:- %s' % (parsed['answer']))
