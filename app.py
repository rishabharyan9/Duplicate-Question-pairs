import streamlit as st
import openai
import helper
from joblib import load

# Initialize OpenAI API
openai.api_key = 'sk-proj-kR5aXzopJymfyUav9PM6T3BlbkFJvkRmukQRTeVsKVHumPgN'

# Load the model
model = load('model.joblib')

# Function to paraphrase text using OpenAI
def paraphrase_text(text):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Or another suitable engine
        prompt=f"Paraphrase the following text: {text}",
        max_tokens=60,  # Adjust based on desired paraphrase length
        n=1,
        stop=None,
        temperature=0.7  # Adjust to control the creativity of the paraphrase
    )
    return response.choices[0].text.strip()

st.header('Duplicate Question Pairs')

# Input fields for the two questions
q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

# When the "Find" button is clicked
if st.button('Find'):
    # Create the feature vector for prediction
    query = helper.query_point_creator(q1, q2)
    # Predict whether the questions are duplicates
    result = model.predict(query)[0]

    if result:
        st.header('Duplicate')
        # If the questions are duplicates, display paraphrased versions
        paraphrased_q1 = paraphrase_text(q1)
        paraphrased_q2 = paraphrase_text(q2)
        st.subheader('Paraphrased Versions:')
        st.write(f"Q1: {paraphrased_q1}")
        st.write(f"Q2: {paraphrased_q2}")
    else:
        st.header('Not Duplicate')

