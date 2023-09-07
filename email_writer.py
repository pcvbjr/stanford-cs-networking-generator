import json
import numpy as np
from numpy.linalg import norm
import os
import re
import time

import openai
import PyPDF2
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff





def generate_resume_notes(resume_file):

    pdf_reader = PyPDF2.PdfReader(resume_file)
    resume = ''
    for page in pdf_reader.pages:
        resume += page.extract_text() + '\n'
        
    def clean_resume(resume):
        resume = resume.replace(' \n', '<newline>')
        resume = resume.replace(' ', '')
        resume = resume.replace('\n', ' ')
        resume = resume.replace('<newline>', '\n')
        return resume

    resume = clean_resume(resume)
    print('RESUME TEXT')
    print('\n'.join(resume.split('\n')[:10])) # print the first 10 lines of the resume
    
    system_prompt = (
        "The user input is text from a resume.\n"
        "Identify the top 3 domains common throughout the resume (for example: front end development, machine learning, data engineering).\n"
        "Fit items in the resume into at least one of these domains.\n"
        "Return a JSON string with keys as the domain names and values as a list of text extracted from the resume that corresponds to that domain.\n"
        "A domain can correspond to one or several extracted text segments.\n"
        "For example: {'Generative AI': ['Developed AI chatbot', 'Built image creation tool with Stable Diffusion'], 'Community': ['Collected 30 thousand books for local schools']}\n"
        "Only include text extracted directly from the resume, do not add any text that is not in the resume except to create the domain names.\n"
        "Order domains from most prevalent in the resume to least.\n"
        "Ignore lines in the resume that do not contribute to one of the top domains.\n"
    )

    user_input = resume
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    
    domains = json.loads(response['choices'][0]['message']['content'])
    for domain, resume_texts in domains.items():
        print(f'##### {domain} #####')
        for text in resume_texts:
            print('--->', text)
        print()
    return domains


@retry(wait=wait_random_exponential(min=0.5, max=5), stop=stop_after_attempt(6))
def embed_completion_with_backoff(text):
    text_embedding = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )['data'][0]['embedding']
    return text_embedding

def get_resume_embeddings(domains):
    resume_embeddings = {} # {DOMAIN: {ARRAY_IDX: {'text': TEXT, 'text_embedding': EMBEDDING_VECTOR}}

    for domain, resume_texts in domains.items():
        domain_data = {}
        array_idx = 0
        for text in resume_texts:
            time.sleep(0.5)
            text_embedding = embed_completion_with_backoff(text)
            domain_data[array_idx] = {
                'text': text,
                'text_embedding': text_embedding
            }
            array_idx += 1
        resume_embeddings[domain] = domain_data
        
    return resume_embeddings


def vec_to_mat_cosine_similarity(vector, matrix):
    p1 = vector.dot(matrix.T)
    p2 = norm(vector) * norm(matrix)
    out = p1 / p2
    return out

@retry(wait=wait_random_exponential(min=0.5, max=5), stop=stop_after_attempt(6))
def chat_completion_with_backoff(system_prompt, user_input):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return response


def write_email(profile, context, email_instruction):
    
    with open('stanford_cs_profiles_full.json', 'r') as f:
        profiles = json.load(f)
    
    receiver = profiles[profile]['name']
    
    system_prompt = f"{email_instruction}\n Include relevant information from the given context. At the end of the email, include a sentence that this was generated using my own custom networking LLM tool."
    user_input = f"Sender: [Sender name] \nReceiver: {receiver} \n"
    for context_item in context:
        resume_text = context_item['resume_text']
        profile_text = context_item['profile_text']
        user_input += f"Context: \n    The Sender achieved the following: {resume_text} \n    The Receiver achieved the following: {profile_text} \n"
    
    print(user_input)
    
    response = chat_completion_with_backoff(system_prompt, user_input)
    
    system_prompt = "Summarize in one sentence what the Receiver has in common with the user."
    summary = chat_completion_with_backoff(system_prompt, user_input)

    return response['choices'][0]['message']['content'], summary['choices'][0]['message']['content'], receiver





def get_email_data(resume_embeddings):
        
    with open('stanford_cs_profiles_150ch_embeddings.json', 'r') as f:
        profile_embeddings = json.load(f)
        
    profile_embedding_mat = np.array([line['text_embedding'] for line in profile_embeddings.values()])

    
    email_data = {} # {PROFILE: [(email 0) {'resume_text': RESUME_TEXT, 'profile_text': PROFILE_TEXT, 'email': EMAIL}, (email 1) ...]}

    for domain, resume_texts in resume_embeddings.items():
        for text_idx, text_details in resume_texts.items():
            resume_text = text_details['text']
            text_embedding = np.array(text_details['text_embedding'])

            print('###', resume_text)
            cs = vec_to_mat_cosine_similarity(text_embedding, profile_embedding_mat)
            for i in range(1):
                max_idx = np.argmax(cs)
                profile_line = profile_embeddings[str(max_idx)]

                profile_text = profile_line['text']
                profile = profile_line['profile']

                print(profile)
                # email = write_email(resume_text, profile_text, profile)

                email_context = {'resume_text': resume_text, 'profile_text': profile_text}
                if profile not in email_data.keys():
                    email_data[profile] = [email_context]
                else:
                    profile_emails = email_data[profile]
                    email_data[profile] = profile_emails + [email_context]
                cs[max_idx] = -99
                
    return email_data


def generate_emails(profile, context, email_instruction):
    if email_instruction is None:
        email_instruction = "Write a short, friendly, professional networking email."
    print('################')
    email, summary, receiver = write_email(profile, context, email_instruction)
    print('### EMAIL ###')
    print(email)
        
    return email, summary, receiver