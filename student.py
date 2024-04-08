from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os

app = Flask(__name__)

# Get your API keys from openai, you will need to create an account.
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
OPENAI_API_KEY = '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'

import pdfplumber
from pytesseract import pytesseract
from pdfminer.high_level import extract_text

# Set up the prompt template for generating use cases
prompt_template = """
You are an education counselor responsible for creating a student profile based on the information provided about the student. The context about the student is as follows: {context}
Based on this information, please generate a comprehensive student profile including the following:
Student Name:
Student Age:
Strengths:
Weaknesses:
Preferred Learning Modes:
Academic Progress:
Recommended Interventions (if any):

Provide a detailed profile covering all the relevant aspects of the student's academic and personal background.
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context"])

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

@app.route('/', methods=['GET', 'POST'])
def index():
    student_profile = ""
    if request.method == 'POST':
        # Get the PDF file from the form
        pdf_file = request.files['pdf_file']

        # Open the PDF file using pdfplumber
        with pdfplumber.open(pdf_file) as pdf:
            num_pages = len(pdf.pages)
            print(f"Number of pages: {num_pages}")

            # Initialize raw_text variable
            raw_text = ''

            # Iterate through pages and extract text
            for i in range(num_pages):
                page = pdf.pages[i]
                text = page.extract_text()
                if text:
                    raw_text += text

        # Split the text into smaller chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)
        print(f"Number of text chunks: {len(texts)}")

        # Create the FAISS index
        docsearch = FAISS.from_texts(texts, embeddings)

        # Set up the LLM and the chain
        llm = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")

        # Generate the student profile
        first_chunk = texts[0]
        query = prompt.format(context=first_chunk)
        docs = docsearch.similarity_search(query)
        student_profile = chain.run(input_documents=docs, question=query)

    return render_template('index8.html', student_profile=student_profile)

if __name__ == '__main__':
    app.run(port=8091,debug=True)