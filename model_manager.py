import torch
from transformers import MarianMTModel, MarianTokenizer
from config import SUPPORTED_LANGUAGES
from device_config import DEVICE

# Model cache to avoid reloading
models = {}
tokenizers = {}

def get_model(pair):
    """
    Load and optimize the translation model for a specific language pair.
    Uses caching to ensure models are only loaded once.
    """
    if pair not in models:
        model_name = SUPPORTED_LANGUAGES.get(pair)
        if not model_name:
            return None, None
            
        print(f"Loading and optimizing model for {pair}...")
        
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # Optimization 1: Use the best available device (GPU/MPS)
        model = model.to(DEVICE)
        
        # Optimization 2: Dynamic Quantization (Only for CPU, significantly speeds up inference)
        if DEVICE == "cpu":
            print(f"Applying dynamic quantization for {pair} on CPU...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        # Optimization 3: Set to evaluation mode
        model.eval()
        
        tokenizers[pair] = tokenizer
        models[pair] = model
        
    return models[pair], tokenizers[pair]
