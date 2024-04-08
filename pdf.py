from flask import Flask, request, render_template
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import requests
import os
from io import BytesIO

# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pdf_url = request.form['pdf_url']

        # Fixed query
        query = "summarize the pdf in 400 words"

        # Fetch the PDF content from the URL
        response = requests.get(pdf_url)
        pdf_bytes = BytesIO(response.content)

        # Create a PdfReader object
        pdf_reader = PdfReader(pdf_bytes)

        # Extract text from the PDF
        raw_text = ''
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        # Split the text into chunks
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        texts = text_splitter.split_text(raw_text)

        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)

        # Load the question-answering chain
        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        # Search for relevant documents and run the chain
        docs = docsearch.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)

        return render_template('index6.html', result=result)

    return render_template('index6.html')

if __name__ == '__main__':
    app.run(port=8099, debug=True)