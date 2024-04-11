import streamlit as st
from pickle import load
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, load_model
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 
import requests
import os
import streamlit as st
from gtts import gTTS
from io import BytesIO

# Function to extract features from an image
def extract_features(filename):
    # Load the VGG16 model
    model = VGG16()
    # Remove the last layer
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # Load and preprocess the image
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    # Extract features
    features = model.predict(image, verbose=0)
    return features

# Function to generate a description for the given image
def generate_description(model, tokenizer, photo, max_length):
    # Seed the generation process
    in_text = 'startseq'
    # Iterate over the sequence length
    for i in range(max_length):
        # Encode the input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad sequences
        sequence = pad_sequences([sequence], maxlen=max_length)
        # Predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # Convert probability to integer
        yhat = np.argmax(yhat)
        # Map integer to word
        word = word_for_id(yhat, tokenizer)
        # Stop if we cannot map the word
        if word is None:
            break
        # Append as input for generating the next word
        in_text += ' ' + word
        # Stop if we predict the end of the sequence
        if word == 'endseq':
            break
        
        query = in_text

        stopwords = ['startseq','endseq'] 
        querywords = query.split()

        resultwords = [word for word in querywords if word.lower() not in stopwords]

        result = ' '.join(resultwords)

        
    return result

# Function to map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return "None"

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")
#img2text

# Load the tokenizer
tokenizer = load(open(r'C:\Users\Viraj\OneDrive\Desktop\foldersss\recognition222\tokenizer.pkl', 'rb'))
# Pre-define the max sequence length
max_length = 34
# Load the model
model = load_model(r'C:\Users\Viraj\OneDrive\Desktop\foldersss\recognition222\model_39.h5')

#story generator
def generate_story(scenario):
    template = '''
    You are text elaborater teller;
    Describe the contents of the image and provide a brief caption or label for the scene, the sentence should be minimum 20 words;

    Content:{scenario}
    elaborated text:
    '''
    
    prompt =PromptTemplate(template=template, input_variables=["scenario"])
    story_llm=LLMChain(llm=OpenAI(openai_api_key="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",model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)
    # story="Viraj"
    print(story)
    return story





# Streamlit web app
def main():
    st.title("Image Captioning App")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = load_img(uploaded_file, target_size=(224, 224))
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        # Generate description on button click
        
        photo = extract_features(uploaded_file)
        scenario = generate_description(model, tokenizer, photo, max_length)
        story = generate_story(scenario)
        
        
        with st.expander("Scenario"):
           st.write(scenario)

        with st.expander("description"):
           st.write(story)

        sound_file = BytesIO()
        tts = gTTS(story, lang='en')
        tts.write_to_fp(sound_file)
        st.audio(sound_file)



if __name__ == "__main__":
    main()