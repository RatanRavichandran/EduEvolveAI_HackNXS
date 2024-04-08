from flask import Flask, render_template, request, flash, redirect
import PyPDF2
import openai

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

openai.api_key = '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'

def get_completion(prompt, model="gpt-3.5-turbo"):
  messages = [{"role": "user", "content": prompt}]
  response = openai.ChatCompletion.create(
     model=model,
     messages=messages,
     temperature=0, # this is the degree of randomness of the model's output
  )
  return response.choices[0].message["content"]

def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file object and returns it as a string.
    """
    pdf_reader = PyPDF2.PdfReader(file)
    extracted_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        extracted_text += page.extract_text()
    return extracted_text

def create_flashcard(text):
    """Generates a question and concise answer from a text chunk."""
    prompt = f"This is a portion of the document content: {text}. Generate a short question and a very concise answer (one sentence). It has to be of format Question: and Answer:"
    answer = get_completion(prompt)

    # Split the answer into two parts based on the first occurrence of "Question:"
    question_start = answer.find("Question:")
    if question_start != -1:
        question_end = answer.find("\n", question_start)
        if question_end == -1:
            question_end = len(answer)
        question = answer[question_start + len("Question:"):question_end].strip()

        # Split the answer into two parts based on the first occurrence of "Answer:"
        answer_start = answer.find("Answer:")
        if answer_start != -1:
            answer_end = answer.find("\n", answer_start)
            if answer_end == -1:
                answer_end = len(answer)
            answer = answer[answer_start + len("Answer:"):answer_end].strip()
        else:
            answer = None
    else:
        question, answer = None, None

    return question, answer

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Read PDF file
            pdf_text = extract_text_from_pdf(file)
            # Generate flashcard from the PDF text
            question, answer = create_flashcard(pdf_text)
            if question and answer:
                flashcard = {'question': question, 'answer': answer}
                return render_template('flashcard.html', flashcard=flashcard)
            else:
                return "Failed to generate flashcard from the provided PDF."

    return render_template('index2.html')

if __name__ == '__main__':
    app.run(port=8093,debug=True)
