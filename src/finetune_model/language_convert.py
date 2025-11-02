
"""
@author: Naveen N G
@date: 30-10-2025
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch



LANGUAGES = {
    'English': 'eng_Latn',
    'Hindi': 'hin_Deva',
    'Kannada': 'kan_Knda',
    'Tamil': 'tam_Taml',
    'Telugu': 'tel_Telu',
    'Malayalam': 'mar_Deva',
    'Marathi': 'mar_Deva',
    'Bengali': 'ben_Beng',
    'Bhojpuri': 'bho_Deva',
    'Gujarati': 'guj_Gujr'
}

class LanguageConverter:

    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"

        # 2. Load Model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Optional: Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "mps"
        self.device = device
        self.model.to(device)

    def set_tokenizer(self, src_lang):
        source_lang_code = LANGUAGES.get(src_lang, 'eng_Latn')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang=source_lang_code)

    def translate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(LANGUAGES.get('English')),
            max_length=150
        )
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text
    
