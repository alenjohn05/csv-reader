# Import necessary libraries and modules
import streamlit as st
import pandas as pd
from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe
import openai

# Set page configuration and title for Streamlit
st.set_page_config(page_title="Chat with CSV", page_icon="📄", layout="wide")

# Add header with title and description
st.markdown(
    '<p style="display:inline-block;font-size:40px;font-weight:bold;">📊csvGPT </p>'
    ' <p style="display:inline-block;font-size:16px;">csvGPT is tool that uses AI-powered '
    'natural language processing to analyze and provide insights on CSV data. Users can '
    'upload CSV files, view the data, and have interactive conversations with the AI model '
    'to obtain valuable information and answers related to the uploaded data <br><br></p>',
    unsafe_allow_html=True
)

# Add an input field for the API key
OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key")

# Check if API key is provided
if OPENAI_API_KEY:
    # Set OpenAI API key
    openai.api_key = OPENAI_API_KEY

    def chat_with_csv(df, prompt):
        llm = OpenAI(api_token=OPENAI_API_KEY, max_tokens=1000, model="gpt-4-1106-preview")
        df = SmartDataframe(df, config={"llm": llm})
        answer = df.chat(prompt)
        print(answer)
        return answer

    input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

    if input_csv is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.info("CSV Uploaded Successfully")
            data = pd.read_csv(input_csv)
            st.dataframe(data, use_container_width=True)

        with col2:
            st.info("Chat Below")
            input_text = st.text_area("Enter your query")

            if input_text is not None:
                if st.button("Chat with CSV"):
                    st.info("Your Query: " + input_text)
                    result = chat_with_csv(data, input_text)
                    st.success(result)
else:
    st.warning("Please enter your OpenAI API Key in the input field above.")
