import os

import streamlit as st
import openai

from email_writer import generate_resume_notes, get_resume_embeddings, get_email_data, generate_emails

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("Stanford CS Networking Generator")
uploaded_file = st.file_uploader("Upload your resume", type=("pdf"))
email_instruction = st.text_input(
    "Give instructions for the output",
    placeholder="Write a short, friendly, professional networking email.",
    disabled=not uploaded_file,
)

if uploaded_file and email_instruction and not openai_api_key:
    st.info("Please add your OpenAI key to continue.")

if uploaded_file and email_instruction and openai_api_key:
    openai.api_key = openai_api_key
    
    # Get key points from resume
    st.write('### Identifying key points from your resume...')
    domains = generate_resume_notes(uploaded_file)
    for domain, resume_texts in domains.items():
        domain_output = f'##### {domain} #####\n'
        for text in resume_texts:
            domain_output += f'- {text}\n'
        st.write(domain_output)
        
    # Get resume embeddings
    resume_embeddings = get_resume_embeddings(domains)
    
    # Compare resume and Stanford profiles
    email_data, profile_names = get_email_data(resume_embeddings)
    
    # Show matches
    st.write(f'### Identified *{len(email_data.keys())}* possible networking opportunities:')
    for (name, profile) in zip(profile_names.values(), email_data.keys()):
        st.write(f'- *[{name}](https://profiles.stanford.edu/{profile})*')
    
    # Generate emails
    st.write('### Generating emails...')
    for profile in email_data.keys():
        email, summary, receiver = generate_emails(profile, email_data[profile], email_instruction)
    
        st.write(f'##### To: {receiver}')
        st.write(f'[Profile Link](https://profiles.stanford.edu/{profile})')
        st.write(f'*Summary: {summary}*')
        st.write(email)