from transformers import (
    MarianMTModel, MarianTokenizer,
    M2M100ForConditionalGeneration, M2M100Tokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
import torch
import os
import PyPDF2
import docx
import openpyxl
from pathlib import Path

class EnhancedDocumentTranslator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_types = {}
        
        # Define model preferences (best to worst)
        self.model_preferences = {
            "nllb": {
                "name": "facebook/nllb-200-distilled-1.3B",
                "quality": "excellent",
                "size": "1.3GB",
                "languages": "200+ languages"
            },
            "m2m100": {
                "name": "facebook/m2m100_418M", 
                "quality": "very_good",
                "size": "418MB",
                "languages": "100+ languages"
            },
            "marian": {
                "name": "Helsinki-NLP/opus-mt-{src}-{tgt}",
                "quality": "good",
                "size": "300MB each",
                "languages": "Specific pairs"
            }
        }
        
        # Language codes for NLLB
        self.nllb_codes = {
            "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn",
            "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
            "ru": "rus_Cyrl", "ja": "jpn_Jpan", "ko": "kor_Hang",
            "zh": "zho_Hans", "ar": "arb_Arab", "hi": "hin_Deva",
            "nl": "nld_Latn", "pl": "pol_Latn", "tr": "tur_Latn",
            "sv": "swe_Latn", "da": "dan_Latn", "no": "nob_Latn"
        }
        
        # Available document translation pairs
        self.available_pairs = {
            # English to other languages
            "en-es": "English to Spanish", "en-fr": "English to French", 
            "en-de": "English to German", "en-it": "English to Italian",
            "en-pt": "English to Portuguese", "en-ru": "English to Russian",
            "en-ja": "English to Japanese", "en-ko": "English to Korean",
            "en-zh": "English to Chinese", "en-ar": "English to Arabic",
            "en-hi": "English to Hindi", "en-nl": "English to Dutch",
            # Other languages to English
            "es-en": "Spanish to English", "fr-en": "French to English",
            "de-en": "German to English", "it-en": "Italian to English",
            "pt-en": "Portuguese to English", "ru-en": "Russian to English",
            "ja-en": "Japanese to English", "ko-en": "Korean to English",
            "zh-en": "Chinese to English", "ar-en": "Arabic to English",
            # Cross-language pairs
            "es-fr": "Spanish to French", "fr-es": "French to Spanish",
            "de-fr": "German to French", "fr-de": "French to German",
            "ja-ko": "Japanese to Korean", "ko-ja": "Korean to Japanese"
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
        """Load Facebook's NLLB model (best quality) - FIXED VERSION"""
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
        """Translate using NLLB model - FIXED VERSION"""
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
            
            # Generate translation - FIXED: Use convert_tokens_to_ids instead of lang_code_to_id
            with torch.no_grad():
                # Get target language token ID correctly
                tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_code)
                
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=1024,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode
            translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translation
            
        except Exception as e:
            print(f"⚠️ NLLB error: {e}")
            # Fallback to M2M-100
            print("🔄 Falling back to M2M-100 model...")
            return self.translate_with_m2m100(text, source_lang, target_lang)
    
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
            print(f"⚠️ M2M-100 error: {e}")
            # Fallback to Marian
            print("🔄 Falling back to Marian model...")
            return self.translate_with_marian(text, source_lang, target_lang)
    
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
    
    def translate_text_smart(self, text, source_lang, target_lang):
        """Smart translation that tries best model first with fallbacks"""
        
        # Try NLLB first if languages are supported
        if source_lang in self.nllb_codes and target_lang in self.nllb_codes:
            print("🤖 Using NLLB model (highest quality)")
            result = self.translate_with_nllb(text, source_lang, target_lang)
            if not result.startswith("Error:"):
                return result
        
        # Try M2M-100 as second choice
        print("🤖 Using M2M-100 model (good quality)")
        result = self.translate_with_m2m100(text, source_lang, target_lang)
        if not result.startswith("Error:"):
            return result
        
        # Finally try Marian
        print("🤖 Using Marian model (baseline)")
        return self.translate_with_marian(text, source_lang, target_lang)
    
    # Document reading methods
    def read_txt_file(self, file_path):
        """Read text from a .txt file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            return "Error: Could not read file with any encoding"
        except Exception as e:
            return f"Error reading file: {e}"
    
    def read_pdf_file(self, file_path):
        """Read text from a PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            return f"Error reading PDF: {e}"
    
    def read_docx_file(self, file_path):
        """Read text from a Word document"""
        try:
            doc = docx.Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
        except Exception as e:
            return f"Error reading Word document: {e}"
    
    def read_document(self, file_path):
        """Read text from various document formats"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return "Error: File not found"
        
        extension = file_path.suffix.lower()
        
        print(f"📖 Reading {extension} file: {file_path.name}")
        
        if extension == '.txt':
            return self.read_txt_file(file_path)
        elif extension == '.pdf':
            return self.read_pdf_file(file_path)
        elif extension in ['.docx', '.doc']:
            return self.read_docx_file(file_path)
        else:
            return f"Error: Unsupported file format '{extension}'. Supported: .txt, .pdf, .docx"
    
    def translate_long_text(self, text, source_lang, target_lang, chunk_size=400):
        """Translate long text by splitting into chunks with smart handling"""
        if len(text) <= chunk_size:
            return self.translate_text_smart(text, source_lang, target_lang)
        
        # Split text into paragraphs first
        paragraphs = text.split('\n\n')
        translated_paragraphs = []
        
        total_paragraphs = len([p for p in paragraphs if p.strip()])
        current_paragraph = 0
        
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                translated_paragraphs.append("")
                continue
                
            current_paragraph += 1
            print(f"🔄 Translating paragraph {current_paragraph}/{total_paragraphs}... ({len(paragraph)} chars)")
            
            # If paragraph is still too long, split by sentences
            if len(paragraph) > chunk_size:
                sentences = paragraph.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) < chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Translate each chunk
                translated_chunks = []
                for j, chunk in enumerate(chunks):
                    print(f"  📝 Chunk {j+1}/{len(chunks)}")
                    result = self.translate_text_smart(chunk, source_lang, target_lang)
                    translated_chunks.append(result)
                
                translated_paragraphs.append(" ".join(translated_chunks))
            else:
                # Translate whole paragraph
                result = self.translate_text_smart(paragraph, source_lang, target_lang)
                translated_paragraphs.append(result)
        
        return "\n\n".join(translated_paragraphs)
    
    def save_translated_document(self, translated_text, original_file_path, source_lang, target_lang):
        """Save translated text to a new file"""
        original_path = Path(original_file_path)
        
        # Create new filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{original_path.stem}_{source_lang}_to_{target_lang}_{timestamp}.txt"
        new_path = original_path.parent / new_name
        
        try:
            with open(new_path, 'w', encoding='utf-8') as file:
                # Add header with translation info
                header = f"# Document Translation\n"
                header += f"# Original: {original_path.name}\n"
                header += f"# Languages: {source_lang} → {target_lang}\n"
                header += f"# Translated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                header += f"# Length: {len(translated_text)} characters\n"
                header += "=" * 50 + "\n\n"
                
                file.write(header + translated_text)
            
            print(f"💾 Translated document saved as: {new_path}")
            return str(new_path)
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return None
    
    def translate_document(self, file_path, source_lang, target_lang):
        """Complete document translation workflow with enhanced features"""
        print(f"\n🌐 Starting Enhanced Document Translation...")
        print(f"📄 File: {Path(file_path).name}")
        print(f"🔄 {source_lang.upper()} → {target_lang.upper()}")
        print(f"🤖 Using multi-model approach (NLLB → M2M100 → Marian)")
        
        # Read document
        text = self.read_document(file_path)
        
        if text.startswith("Error:"):
            return text
        
        if not text.strip():
            return "Error: Document appears to be empty or unreadable"
        
        print(f"📊 Document length: {len(text):,} characters")
        # FIXED: Move the split operation outside the f-string
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        print(f"📄 Estimated paragraphs: {paragraph_count}")
        
        # Translate text
        print("\n🚀 Starting translation process...")
        translated_text = self.translate_long_text(text, source_lang, target_lang)
        
        if translated_text.startswith("Error:"):
            return translated_text
        
        # Save translated document
        saved_path = self.save_translated_document(translated_text, file_path, source_lang, target_lang)
        
        if saved_path:
            # Show preview
            preview = translated_text[:300] + "..." if len(translated_text) > 300 else translated_text
            print(f"\n📋 Translation Preview:")
            print(f"─" * 40)
            print(preview)
            print(f"─" * 40)
            
            return f"✅ Translation completed successfully!\n📁 Saved to: {saved_path}\n📊 Translated: {len(translated_text):,} characters"
        else:
            return "⚠️ Translation completed but failed to save file"

def main():
    """Main interface - Document translation focused"""
    translator = EnhancedDocumentTranslator()
    
    print("📚 Enhanced Document Translation Tool")
    print("=" * 60)
    print("🎯 PRIMARY FEATURE: Complete Document Translation")
    print("🤖 Multi-Model Approach: NLLB → M2M-100 → Marian")
    print("📄 Supported formats: .txt, .pdf, .docx")
    print("🌐 200+ languages supported")
    print("=" * 60)
    
    while True:
        print("\n" + "─" * 60)
        print("🎯 MAIN OPTIONS:")
        print("1. 📄 TRANSLATE DOCUMENT (Recommended)")
        print("2. 📝 Quick text translation")
        print("3. 🔍 Compare translation models")
        print("4. 🌐 Show available languages")
        print("5. 🤖 Show model information")
        print("6. 🚪 Exit")
        
        choice = input(f"\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            # MAIN FEATURE: Document translation
            print("\n📄 DOCUMENT TRANSLATION")
            print("─" * 30)
            
            file_path = input("Enter full path to document: ").strip().strip('"')
            
            if not file_path:
                print("❌ Please provide a file path")
                continue
            
            print(f"\n🌐 Popular language pairs:")
            popular_pairs = ["en-es", "en-fr", "en-de", "es-en", "fr-en", "ko-en", "ja-en", "zh-en"]
            for pair in popular_pairs:
                desc = translator.available_pairs.get(pair, "")
                print(f"  {pair}: {desc}")
            
            source_lang = input(f"\nSource language code (e.g., 'en', 'ko', 'ja'): ").strip().lower()
            target_lang = input("Target language code (e.g., 'en', 'es', 'fr'): ").strip().lower()
            
            if source_lang and target_lang:
                print(f"\n🚀 Starting translation process...")
                result = translator.translate_document(file_path, source_lang, target_lang)
                print(f"\n{result}")
            else:
                print("❌ Please provide both source and target language codes")
        
        elif choice == "2":
            # Quick text translation
            print("\n📝 QUICK TEXT TRANSLATION")
            print("─" * 30)
            
            text = input("Enter text to translate: ").strip()
            source_lang = input("Source language (en/es/fr/de/ko/ja/etc): ").strip().lower()
            target_lang = input("Target language (en/es/fr/de/ko/ja/etc): ").strip().lower()
            
            if text and source_lang and target_lang:
                print(f"\n🔄 Translating...")
                result = translator.translate_text_smart(text, source_lang, target_lang)
                print(f"\n📝 Original: {text}")
                print(f"🌐 Translation: {result}")
        
        elif choice == "3":
            # Model comparison
            print("\n🔍 MODEL COMPARISON")
            print("─" * 30)
            
            text = input("Enter text to compare (short text recommended): ").strip()
            source_lang = input("Source language: ").strip().lower()
            target_lang = input("Target language: ").strip().lower()
            
            if text and source_lang and target_lang:
                print(f"\n🔍 Comparing models for: '{text}'")
                print(f"🌐 {source_lang} → {target_lang}")
                print("=" * 50)
                
                # Try each model
                models_to_try = [
                    ("NLLB (Best)", "nllb"),
                    ("M2M-100 (Good)", "m2m100"), 
                    ("Marian (Baseline)", "marian")
                ]
                
                for model_name, model_key in models_to_try:
                    print(f"\n🤖 Testing {model_name}...")
                    if model_key == "nllb":
                        result = translator.translate_with_nllb(text, source_lang, target_lang)
                    elif model_key == "m2m100":
                        result = translator.translate_with_m2m100(text, source_lang, target_lang)
                    else:
                        result = translator.translate_with_marian(text, source_lang, target_lang)
                    
                    print(f"  {model_name}: {result}")
        
        elif choice == "4":
            print("\n🌐 AVAILABLE LANGUAGE PAIRS:")
            print("=" * 50)
            for code, description in translator.available_pairs.items():
                print(f"{code}: {description}")
            print("=" * 50)
            print(f"Total supported pairs: {len(translator.available_pairs)}")
            
        elif choice == "5":
            translator.show_model_info()
            
        elif choice == "6":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()