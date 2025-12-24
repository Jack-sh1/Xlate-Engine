from flask import Flask, render_template
from scss_compiler import compile_scss
from server_config import HOST, PORT, DEBUG
from api_handlers import translate_api

# Run SCSS compilation
compile_scss()

# --- Flask Web Application ---

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/translate', methods=['POST'])
def translate():
    return translate_api()
