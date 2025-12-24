import torch
from device_config import DEVICE
from model_manager import get_model

def real_translate(text, pair):
    """
    Core translation logic. Handles direct translation and pivot translation via English.
    """
    if not text.strip():
        return ""
    
    # Check if we have a direct model
    model, tokenizer = get_model(pair)
    if model:
        # Optimization 4: Use inference_mode for faster computation
        with torch.inference_mode():
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(DEVICE)
            translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    
    # If no direct model, try pivot translation via English
    # Example: zh-es -> zh-en then en-es
    if '-' not in pair:
        return "Error: Invalid language pair format."
        
    src, tgt = pair.split('-')
    if src != 'en' and tgt != 'en':
        print(f"No direct model for {pair}, trying pivot translation via English...")
        # Step 1: src -> en
        en_text = real_translate(text, f"{src}-en")
        if en_text.startswith("Error:"):
            return en_text
        
        # Step 2: en -> tgt
        return real_translate(en_text, f"en-{tgt}")
    
    return "Error: Unsupported language pair."
