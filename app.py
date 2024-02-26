import os 
import requests
import streamlit as st
# from dotenv import find_dotenv, load_dotenv
# from langchain.llms import OpenAI
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

# load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN =  os.environ['HUGGINGFACEHUB_API_TOKEN']

def img2text(url):
    img_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = img_to_text(url)
    print(text)
    return text

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo-1b")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/cosmo-1b").to('cpu')
def textGenerator(scenario):
    prompt = f"generate a story using a about: {scenario[0]['generated_text']}"
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to('cpu')
    output = model.generate(
        **inputs,
        max_length=200,  # Adjust the desired length of the story
        do_sample=True,
        temperature=0.6,  # Control randomness (higher values make it more random)
        top_p=0.95,  # Control diversity (lower values make it more focused)
        repetition_penalty=1.2,  # Penalize repeated phrases
    )
    text = tokenizer.decode(output[0]) 
    print(text)
    return text
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-tha"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payload = {
        'inputs':message + message
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    # Audio(response.content)

    with open('audio.flac', 'wb') as file:
            file.write(response.content)


def main():
    st.set_page_config(page_title="img 2 audio sotry", page_icon="ii")
    st.header("Turn img into audio story")
    uploaded_file = st.file_uploader("Chose an image....", type='jpg')
    if uploaded_file:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        scenario = img2text('photo.jpg')
        story = textGenerator(scenario)
        story = story.split(']')[-1]
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        st.audio("audio.flac")

if __name__ == "__main__":
    main()