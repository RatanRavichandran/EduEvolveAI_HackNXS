from flask import Flask, render_template, request
import openai
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI

app = Flask(__name__)

# Configure your OpenAI API key
OPENAI_api_key = "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"

template = """
You are an expert quiz maker for technical fields. Let's think step by step and
create a quiz with {num_questions} {quiz_type} questions about the following concept/content: {quiz_context}.

The format of the quiz could be one of the following:
... (rest of the template)
"""

prompt = PromptTemplate.from_template(template)

def generate_quiz(num_questions, quiz_context, quiz_type):
    chain = LLMChain(llm=ChatOpenAI(openai_api_key="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", temperature=0.0), prompt=prompt)
    quiz_response = chain.run(num_questions=num_questions, quiz_type=quiz_type, quiz_context=quiz_context)
    return quiz_response

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_questions = int(request.form['num_questions'])
        quiz_context = request.form['quiz_context']
        quiz_type = request.form['quiz_type']
        quiz_response = generate_quiz(num_questions, quiz_context, quiz_type)
        return render_template('index4.html', quiz_response=quiz_response)
    return render_template('index4.html')

if __name__ == '__main__':
    app.run(port=8095,debug=True)