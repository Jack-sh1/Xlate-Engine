# Machine Translation Model Configuration
# Format: 'source-target': 'model_name'

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

# Optional: List version for UI iteration if needed
LANGUAGE_LIST = [
    {'code': 'zh', 'name': '中文 (Chinese)'},
    {'code': 'en', 'name': '英文 (English)'},
    {'code': 'es', 'name': '西班牙语 (Spanish)'},
    {'code': 'de', 'name': '德语 (German)'},
    {'code': 'fr', 'name': '法语 (French)'}
]
