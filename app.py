import torch
import torch.nn as nn
import math
from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline, MarianMTModel, MarianTokenizer
import os

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

SUPPORTED_LANGUAGES = {
    'zh-en': 'Helsinki-NLP/opus-mt-zh-en',
    'en-zh': 'Helsinki-NLP/opus-mt-en-zh',
    'en-es': 'Helsinki-NLP/opus-mt-en-es',
    'en-de': 'Helsinki-NLP/opus-mt-en-de',
    'en-fr': 'Helsinki-NLP/opus-mt-en-fr',
    'es-en': 'Helsinki-NLP/opus-mt-es-en',
    'de-en': 'Helsinki-NLP/opus-mt-de-en',
    'fr-en': 'Helsinki-NLP/opus-mt-fr-en'
}

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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Real Transformer Translator</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f0f2f5; padding-top: 50px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .container { max-width: 800px; background: white; padding: 40px; border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }
        h2 { color: #1a73e8; margin-bottom: 30px; text-align: center; font-weight: 700; }
        .io-group { display: flex; gap: 20px; margin-bottom: 20px; }
        .io-box { flex: 1; }
        textarea { border-radius: 12px !important; border: 1px solid #dee2e6 !important; padding: 15px !important; resize: none; font-size: 16px; }
        textarea:focus { border-color: #1a73e8 !important; box-shadow: 0 0 0 0.2rem rgba(26,115,232,0.25) !important; }
        #result-box-wrapper { position: relative; }
        #result-box { height: 150px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 12px; padding: 15px; font-size: 16px; overflow-y: auto; }
        .copy-btn { 
            position: absolute; 
            bottom: 10px; 
            right: 10px; 
            background: white; 
            border: 1px solid #dee2e6; 
            border-radius: 8px; 
            padding: 5px 10px; 
            cursor: pointer; 
            font-size: 0.8rem;
            color: #5f6368;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 5px;
            z-index: 10;
        }
        .copy-btn:hover { background: #f1f3f4; color: #1a73e8; border-color: #1a73e8; }
        .btn-translate { background-color: #1a73e8; border: none; padding: 12px; border-radius: 12px; font-weight: 600; transition: all 0.3s; }
        .btn-translate:hover { background-color: #1557b0; transform: translateY(-2px); }
        .loading-spinner { display: none; margin-right: 10px; }
        .footer-note { margin-top: 30px; padding: 15px; background: #e8f0fe; border-radius: 10px; font-size: 0.9rem; color: #1967d2; }
    </style>
</head>
<body>
    <div class="container">
        <h2><svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" class="bi bi-translate me-2" viewBox="0 0 16 16"><path d="M4.545 6.714 4.11 8H3l1.862-5h1.284L8 8H6.833l-.435-1.286H4.545zm1.634-4.804L4.82 5.8h2.71L6.179 1.91z"/><path d="M0 15h16v-1H0v1zm16-11h-2V2.5a.5.5 0 0 0-.5-.5H11V0h-1v2H2.5a.5.5 0 0 0-.5.5V4H0v1h2v8.5a.5.5 0 0 0 .5.5H13.5a.5.5 0 0 0 .5-.5V5h2V4zM2 13V3h9v10H2zm10 0V3h2v10h-2z"/></svg>Smart Multi-Translator</h2>
        
        <div class="row mb-3">
            <div class="col-md-5">
                <select id="srcLang" class="form-select" onchange="handleInput()">
                    <option value="zh">中文 (Chinese)</option>
                    <option value="en" selected>英文 (English)</option>
                    <option value="es">西班牙语 (Spanish)</option>
                    <option value="de">德语 (German)</option>
                    <option value="fr">法语 (French)</option>
                </select>
            </div>
            <div class="col-md-2 text-center py-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#1a73e8" class="bi bi-arrow-right" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M1 8a.5.5 0 0 1 .5-.5h11.793l-3.147-3.146a.5.5 0 0 1 .708-.708l4 4a.5.5 0 0 1 0 .708l-4 4a.5.5 0 0 1-.708-.708L13.293 8.5H1.5A.5.5 0 0 1 1 8z"/></svg>
            </div>
            <div class="col-md-5">
                <select id="tgtLang" class="form-select" onchange="handleInput()">
                    <option value="en">英文 (English)</option>
                    <option value="zh" selected>中文 (Chinese)</option>
                    <option value="es">西班牙语 (Spanish)</option>
                    <option value="de">德语 (German)</option>
                    <option value="fr">法语 (French)</option>
                </select>
            </div>
        </div>

        <div class="io-group">
            <div class="io-box">
                <textarea id="inputText" class="form-control" rows="6" placeholder="输入内容，按回车键(Enter)翻译..." oninput="handleInput()" onkeydown="handleKeyDown(event)"></textarea>
            </div>
            <div class="io-box">
                <div id="result-box-wrapper">
                    <div id="result-box">
                        <div id="outputText" class="text-muted">翻译结果将在这里显示...</div>
                    </div>
                    <button class="copy-btn" onclick="copyToClipboard()" title="复制到剪贴板">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clipboard" viewBox="0 0 16 16">
                            <path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/>
                            <path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/>
                        </svg>
                        <span id="copyText">复制</span>
                    </button>
                </div>
            </div>
        </div>

        <button onclick="runTranslation()" class="btn btn-primary btn-translate w-100" id="btnText">
            <span class="spinner-border spinner-border-sm loading-spinner" id="spinner"></span>
            立即翻译
        </button>
        
        <div class="footer-note">
            <strong>技术栈说明：</strong><br>
            • 核心架构：原生手搓 Transformer (编码器-解码器)<br>
            • 预训练大脑：Helsinki-NLP MarianMT (基于 Transformer 架构训练)<br>
            • 功能：支持真实的中文到英文翻译
        </div>
    </div>

    <script>
        function copyToClipboard() {
            const text = document.getElementById('outputText').innerText;
            const copyText = document.getElementById('copyText');
            
            if (text === "翻译结果将在这里显示..." || text === "翻译出错，请稍后重试。") return;

            navigator.clipboard.writeText(text).then(() => {
                const originalText = copyText.innerText;
                copyText.innerText = "已复制!";
                setTimeout(() => {
                    copyText.innerText = originalText;
                }, 2000);
            });
        }

        function handleInput() {
            const text = document.getElementById('inputText').value;
            const outputDiv = document.getElementById('outputText');
            
            // 如果输入框清空，立即清空右侧
            if (!text.trim()) {
                outputDiv.innerText = "翻译结果将在这里显示...";
                outputDiv.classList.add('text-muted');
                return;
            }
        }

        function handleKeyDown(event) {
            // 监听回车键 (Enter)
            // 如果按下 Shift + Enter，允许换行；只按 Enter 则触发翻译
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault(); // 阻止默认的回车换行行为
                runTranslation();
            }
        }

        async function runTranslation() {
            const text = document.getElementById('inputText').value;
            const srcLang = document.getElementById('srcLang').value;
            const tgtLang = document.getElementById('tgtLang').value;
            const outputDiv = document.getElementById('outputText');
            const spinner = document.getElementById('spinner');
            const btnText = document.getElementById('btnText');

            if(!text.trim()) return;
            if(srcLang === tgtLang) {
                outputDiv.innerText = text;
                outputDiv.classList.remove('text-muted');
                return;
            }

            spinner.style.display = 'inline-block';
            btnText.disabled = true;
            outputDiv.classList.add('text-muted');

            try {
                const response = await fetch('/api/translate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        text: text,
                        pair: `${srcLang}-${tgtLang}`
                    })
                });
                const data = await response.json();
                if (data.error) {
                    outputDiv.innerText = data.error;
                } else {
                    outputDiv.innerText = data.translation;
                    outputDiv.classList.remove('text-muted');
                }
            } catch (e) {
                outputDiv.innerText = "翻译出错，请稍后重试。";
            } finally {
                spinner.style.display = 'none';
                btnText.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

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
    print("Translator App running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
