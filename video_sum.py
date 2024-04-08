YOUTUBE_DATA_API_KEY = 'AIzaSyDWSmR2qRaTviw739lZObCIKUM9FREHN2U'
GEMINI_API_KEY = 'AIzaSyCQeU7Nn1LJawGpnaFmkjtuTa0-84H4zhA'

from flask import Flask, render_template, request
import googleapiclient.discovery
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import google.generativeai as genai
from urllib.parse import urlparse, parse_qs


app = Flask(__name__, static_folder='static')


youtube = googleapiclient.discovery.build(serviceName='youtube', version='v3', developerKey=YOUTUBE_DATA_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
genai_model = genai.GenerativeModel('gemini-pro')

def get_summary_from_youtube_link(youtube_link, extract_prompt="Summarize video content and give Notes "):
    video_id = extract_video_id(youtube_link)

    if video_id:
        transcript = get_transcript(video_id)
        summary, _, _ = get_ai_extract(extract_prompt, transcript)
        
        # Remove ** from the summary
        summary = summary.replace('**', '')
        
        # Split summary into sentences
        sentences = summary.split('. ')
        
        # Format sentences as bullet points
        bullet_points = '\n'.join(['- ' + sentence for sentence in sentences if sentence.strip()])

        return bullet_points
    else:
        return "Invalid YouTube link. Please provide a valid link."

def get_transcript(video_id, languages=['en', 'en-US', 'en-GB']):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    transcript = TextFormatter().format_transcript(transcript)
    return transcript

def get_ai_extract(prompt, text):
    response = genai_model.generate_content(prompt + text, stream=False)
    return response.text, response.prompt_feedback, response.candidates

def extract_video_id(youtube_link):
    parsed_url = urlparse(youtube_link)
    video_id = parse_qs(parsed_url.query).get('v')
    return video_id[0] if video_id else None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    youtube_link = request.form['youtube_link']
    summary = get_summary_from_youtube_link(youtube_link)
    return render_template('result.html', youtube_link=youtube_link, summary=summary)

if __name__ == '__main__':
    app.run(port=8092)