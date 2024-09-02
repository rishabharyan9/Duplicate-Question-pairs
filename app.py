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
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Use the appropriate model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that paraphrases text."},
            {"role": "user", "content": f"Paraphrase the following text: {text}"}
        ],
        max_tokens=60,
        temperature=0.7,
    )
    return response.choices[0].message['content'].strip()

# Example usage:
# paraphrased_text = paraphrase_text("Your original text here.")
# print(paraphrased_text)

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

