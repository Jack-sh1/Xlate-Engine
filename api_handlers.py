from flask import request, jsonify
from translator_service import real_translate

def translate_api():
    """
    API handler for translation requests.
    Processes JSON input and returns translation results.
    """
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
