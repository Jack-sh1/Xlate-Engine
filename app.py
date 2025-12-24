import torch
import torch.nn as nn
import math
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, MarianMTModel, MarianTokenizer
import os
from scss_compiler import compile_scss
from config import SUPPORTED_LANGUAGES
from server_config import HOST, PORT, DEBUG

# Run SCSS compilation
compile_scss()

# --- 1. Educational Content: Raw Transformer Architecture (Optional/Reference) ---
# (Keeping this here so you can still see the hand-rolled core code)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc(context)

class Transformer(nn.Module):
    # (Simplified version for reference)
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128):
        super(Transformer, self).__init__()
        self.emb = nn.Embedding(src_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.mha = MultiHeadAttention(d_model, 4)
        self.out = nn.Linear(d_model, tgt_vocab_size)
    def forward(self, x): return self.out(self.mha(x,x,x))

# --- 2. Production Ready: Multi-language Translation Model ---

# Determine best device (MPS for Mac, CUDA for NVIDIA, CPU as fallback)
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
print(f"Using device: {device}")

# Model cache to avoid reloading
models = {}
tokenizers = {}



def get_model(pair):
    if pair not in models:
        model_name = SUPPORTED_LANGUAGES.get(pair)
        if not model_name: return None, None
        print(f"Loading and optimizing model for {pair}...")
        
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # Optimization 1: Use the best available device (GPU/MPS)
        model = model.to(device)
        
        # Optimization 2: Dynamic Quantization (Only for CPU, significantly speeds up inference)
        if device == "cpu":
            print(f"Applying dynamic quantization for {pair} on CPU...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        # Optimization 3: Set to evaluation mode
        model.eval()
        
        tokenizers[pair] = tokenizer
        models[pair] = model
        
    return models[pair], tokenizers[pair]

def real_translate(text, pair):
    if not text.strip(): return ""
    
    # Check if we have a direct model
    model, tokenizer = get_model(pair)
    if model:
        # Optimization 4: Use inference_mode for faster computation
        with torch.inference_mode():
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
            translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    
    # If no direct model, try pivot translation via English
    # Example: zh-es -> zh-en then en-es
    src, tgt = pair.split('-')
    if src != 'en' and tgt != 'en':
        print(f"No direct model for {pair}, trying pivot translation via English...")
        # Step 1: src -> en
        en_text = real_translate(text, f"{src}-en")
        if en_text.startswith("Error:"): return en_text
        
        # Step 2: en -> tgt
        return real_translate(en_text, f"en-{tgt}")
    
    return "Error: Unsupported language pair."

# --- 3. Flask Web Application ---

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/translate', methods=['POST'])
def translate_api():
    data = request.json
    text = data.get('text', '')
    pair = data.get('pair', 'zh-en')
    
    if not text:
        return jsonify({"translation": ""})
    
    try:
        translation = real_translate(text, pair)
        return jsonify({"translation": translation})
    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({"error": f"暂不支持该语种组合或模型加载失败"}), 500

if __name__ == '__main__':
    print(f"Translator App running at http://127.0.0.1:{PORT}")
    # Note: Running on port 80 requires sudo/root privileges on most systems
    app.run(debug=DEBUG, host=HOST, port=PORT)
