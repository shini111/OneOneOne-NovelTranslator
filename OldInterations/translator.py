from transformers import (
    MarianMTModel, MarianTokenizer,
    M2M100ForConditionalGeneration, M2M100Tokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
import torch

class AdvancedTranslator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_types = {}
        
        # Define model preferences (best to worst)
        self.model_preferences = {
            # Facebook's NLLB - Best multilingual model
            "nllb": {
                "name": "facebook/nllb-200-distilled-1.3B",
                "quality": "excellent",
                "size": "1.3GB",
                "languages": "200+ languages"
            },
            # Facebook's M2M-100 - Very good multilingual
            "m2m100": {
                "name": "facebook/m2m100_418M", 
                "quality": "very_good",
                "size": "418MB",
                "languages": "100+ languages"
            },
            # Helsinki Opus - Good baseline
            "marian": {
                "name": "Helsinki-NLP/opus-mt-{src}-{tgt}",
                "quality": "good",
                "size": "300MB each",
                "languages": "Specific pairs"
            }
        }
        
        # Language codes for NLLB (different from Helsinki)
        self.nllb_codes = {
            "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn",
            "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
            "ru": "rus_Cyrl", "ja": "jpn_Jpan", "ko": "kor_Hang",
            "zh": "zho_Hans", "ar": "arb_Arab", "hi": "hin_Deva"
        }
        
    def show_model_info(self):
        """Display information about available models"""
        print("\n🤖 Available Translation Models:")
        print("=" * 60)
        for model_id, info in self.model_preferences.items():
            print(f"{model_id.upper()}:")
            print(f"  Model: {info['name']}")
            print(f"  Quality: {info['quality']}")
            print(f"  Size: {info['size']}")
            print(f"  Languages: {info['languages']}")
            print()
    
    def load_nllb_model(self):
        """Load Facebook's NLLB model (best quality)"""
        model_name = "facebook/nllb-200-distilled-1.3B"
        print(f"🔄 Loading NLLB model (this is large - 1.3GB)...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            self.tokenizers["nllb"] = tokenizer
            self.models["nllb"] = model
            self.model_types["nllb"] = "nllb"
            
            print("✅ NLLB model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading NLLB: {e}")
            return False
    
    def load_m2m100_model(self):
        """Load Facebook's M2M-100 model (good quality, smaller)"""
        model_name = "facebook/m2m100_418M"
        print(f"🔄 Loading M2M-100 model...")
        
        try:
            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            model = M2M100ForConditionalGeneration.from_pretrained(model_name)
            
            self.tokenizers["m2m100"] = tokenizer
            self.models["m2m100"] = model
            self.model_types["m2m100"] = "m2m100"
            
            print("✅ M2M-100 model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading M2M-100: {e}")
            return False
    
    def load_marian_model(self, source_lang, target_lang):
        """Load Helsinki Marian model (baseline)"""
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        key = f"marian_{source_lang}_{target_lang}"
        
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            self.tokenizers[key] = tokenizer
            self.models[key] = model
            self.model_types[key] = "marian"
            
            return True
        except Exception as e:
            print(f"❌ Error loading Marian model: {e}")
            return False
    
    def translate_with_nllb(self, text, source_lang, target_lang):
        """Translate using NLLB model"""
        if "nllb" not in self.models:
            if not self.load_nllb_model():
                return "Error: Could not load NLLB model"
        
        tokenizer = self.tokenizers["nllb"]
        model = self.models["nllb"]
        
        # Convert language codes
        src_code = self.nllb_codes.get(source_lang, source_lang)
        tgt_code = self.nllb_codes.get(target_lang, target_lang)
        
        try:
            # Set source language
            tokenizer.src_lang = src_code
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
                    max_length=1024,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode
            translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translation
            
        except Exception as e:
            return f"NLLB translation error: {e}"
    
    def translate_with_m2m100(self, text, source_lang, target_lang):
        """Translate using M2M-100 model"""
        if "m2m100" not in self.models:
            if not self.load_m2m100_model():
                return "Error: Could not load M2M-100 model"
        
        tokenizer = self.tokenizers["m2m100"]
        model = self.models["m2m100"]
        
        try:
            # Set source language
            tokenizer.src_lang = source_lang
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.get_lang_id(target_lang),
                    max_length=1024,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode
            translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translation
            
        except Exception as e:
            return f"M2M-100 translation error: {e}"
    
    def translate_with_marian(self, text, source_lang, target_lang):
        """Translate using Marian model"""
        key = f"marian_{source_lang}_{target_lang}"
        
        if key not in self.models:
            if not self.load_marian_model(source_lang, target_lang):
                return "Error: Could not load Marian model"
        
        tokenizer = self.tokenizers[key]
        model = self.models[key]
        
        try:
            inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
            
        except Exception as e:
            return f"Marian translation error: {e}"
    
    def translate_with_best_model(self, text, source_lang, target_lang, preferred_model="auto"):
        """Translate using the best available model"""
        
        if preferred_model == "nllb" or (preferred_model == "auto" and source_lang in self.nllb_codes and target_lang in self.nllb_codes):
            print("🤖 Using NLLB model (highest quality)")
            return self.translate_with_nllb(text, source_lang, target_lang)
        
        elif preferred_model == "m2m100" or preferred_model == "auto":
            print("🤖 Using M2M-100 model (good quality)")
            return self.translate_with_m2m100(text, source_lang, target_lang)
        
        else:
            print("🤖 Using Marian model (baseline)")
            return self.translate_with_marian(text, source_lang, target_lang)
    
    def compare_models(self, text, source_lang, target_lang):
        """Compare translation quality across all models"""
        print(f"\n🔍 Comparing models for: '{text}'")
        print(f"🌐 {source_lang} → {target_lang}")
        print("=" * 50)
        
        results = {}
        
        # Try NLLB
        print("Testing NLLB...")
        results["NLLB (Best)"] = self.translate_with_nllb(text, source_lang, target_lang)
        
        # Try M2M-100
        print("Testing M2M-100...")
        results["M2M-100 (Good)"] = self.translate_with_m2m100(text, source_lang, target_lang)
        
        # Try Marian
        print("Testing Marian...")
        results["Marian (Baseline)"] = self.translate_with_marian(text, source_lang, target_lang)
        
        # Display results
        for model_name, translation in results.items():
            print(f"\n{model_name}:")
            print(f"  {translation}")
        
        return results

def main():
    """Interactive interface for advanced translation"""
    translator = AdvancedTranslator()
    
    print("🚀 Advanced Translation Tool")
    print("=" * 50)
    print("Features multiple high-quality models:")
    translator.show_model_info()
    
    while True:
        print("\n" + "─" * 50)
        print("Options:")
        print("1. Translate with best model")
        print("2. Compare all models")
        print("3. Choose specific model")
        print("4. Show model info")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            text = input("\nEnter text to translate: ").strip()
            source_lang = input("Source language (en/es/fr/de/etc): ").strip().lower()
            target_lang = input("Target language (en/es/fr/de/etc): ").strip().lower()
            
            if text and source_lang and target_lang:
                result = translator.translate_with_best_model(text, source_lang, target_lang)
                print(f"\n📝 Original: {text}")
                print(f"🌐 Translation: {result}")
        
        elif choice == "2":
            text = input("\nEnter text to compare: ").strip()
            source_lang = input("Source language: ").strip().lower()
            target_lang = input("Target language: ").strip().lower()
            
            if text and source_lang and target_lang:
                translator.compare_models(text, source_lang, target_lang)
        
        elif choice == "3":
            print("\nAvailable models:")
            print("1. NLLB (highest quality)")
            print("2. M2M-100 (good quality)")
            print("3. Marian (baseline)")
            
            model_choice = input("Choose model (1-3): ").strip()
            text = input("Enter text: ").strip()
            source_lang = input("Source language: ").strip().lower()
            target_lang = input("Target language: ").strip().lower()
            
            model_map = {"1": "nllb", "2": "m2m100", "3": "marian"}
            
            if model_choice in model_map and text and source_lang and target_lang:
                result = translator.translate_with_best_model(text, source_lang, target_lang, model_map[model_choice])
                print(f"\n📝 Original: {text}")
                print(f"🌐 Translation: {result}")
        
        elif choice == "4":
            translator.show_model_info()
        
        elif choice == "5":
            print("👋 Goodbye!")
            break

if __name__ == "__main__":
    main()