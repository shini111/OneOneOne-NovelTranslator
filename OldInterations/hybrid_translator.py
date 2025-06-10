from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    MarianMTModel, MarianTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer,
    BitsAndBytesConfig
)
import torch
import os
import PyPDF2
import docx
from pathlib import Path
from datetime import datetime
import gc
import re

class AdvancedHybridTranslator:
    def __init__(self):
        self.translation_models = {}
        self.llm_model = None
        self.llm_tokenizer = None
        self.current_llm = None
        
        # Enhanced language codes for different models
        self.nllb_codes = {
            "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn",
            "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
            "ru": "rus_Cyrl", "ja": "jpn_Jpan", "ko": "kor_Hang",
            "zh": "zho_Hans", "ar": "arb_Arab", "hi": "hin_Deva",
            "nl": "nld_Latn", "pl": "pol_Latn", "tr": "tur_Latn"
        }
        
        # Available local LLM models (ordered by quality/capability)
        self.available_llms = {
            "deepseek-coder": {
                "model": "deepseek-ai/deepseek-coder-1.3b-instruct",
                "size": "1.3B parameters",
                "good_for": "Code, structured text, technical translation",
                "ram_needed": "4GB"
            },
            "deepseek-llm": {
                "model": "deepseek-ai/deepseek-llm-7b-chat",
                "size": "7B parameters", 
                "good_for": "General translation, context understanding",
                "ram_needed": "8GB"
            },
            "qwen-chat": {
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "size": "3B parameters",
                "good_for": "Multilingual, excellent for Asian languages",
                "ram_needed": "6GB"
            },
            "qwen-large": {
                "model": "Qwen/Qwen2.5-7B-Instruct", 
                "size": "7B parameters",
                "good_for": "Best multilingual performance",
                "ram_needed": "12GB"
            },
            "mistral": {
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "size": "7B parameters",
                "good_for": "General purpose, good reasoning",
                "ram_needed": "10GB"
            }
        }
        
        # Korean-specific optimizations
        self.korean_patterns = {
            "honorifics": ["님", "씨", "선생", "박사", "교수"],
            "formal_endings": ["습니다", "ㅂ니다", "세요", "하세요"],
            "informal_endings": ["해", "야", "어", "네"]
        }
        
    def show_available_llms(self):
        """Display available local LLM models"""
        print("\n🤖 Available Local LLM Models:")
        print("=" * 70)
        for model_id, info in self.available_llms.items():
            print(f"{model_id.upper()}:")
            print(f"  Model: {info['model']}")
            print(f"  Size: {info['size']}")
            print(f"  Good for: {info['good_for']}")
            print(f"  RAM needed: {info['ram_needed']}")
            print()
    
    def load_llm_model(self, model_choice="qwen-chat"):
        """Load local LLM model with optimization"""
        if model_choice not in self.available_llms:
            print(f"❌ Unknown model: {model_choice}")
            return False
            
        model_name = self.available_llms[model_choice]["model"]
        print(f"🔄 Loading LLM: {model_name}")
        print(f"⚠️ This requires {self.available_llms[model_choice]['ram_needed']} RAM")
        
        try:
            # Configure for efficient loading
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model with quantization for efficiency
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Set padding token
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                
            self.current_llm = model_choice
            print(f"✅ LLM {model_choice} loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading LLM: {e}")
            print("💡 Try a smaller model or ensure you have enough RAM/VRAM")
            return False
    
    def load_translation_model(self, model_type, source_lang, target_lang):
        """Load specialized translation model"""
        if model_type == "nllb":
            return self.load_nllb_model()
        elif model_type == "m2m100":
            return self.load_m2m100_model() 
        elif model_type == "marian":
            return self.load_marian_model(source_lang, target_lang)
        
    def load_nllb_model(self):
        """Load NLLB model"""
        if "nllb" in self.translation_models:
            return True
            
        model_name = "facebook/nllb-200-distilled-1.3B"
        print(f"🔄 Loading NLLB model...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            self.translation_models["nllb"] = {
                "tokenizer": tokenizer,
                "model": model,
                "type": "nllb"
            }
            
            print("✅ NLLB model loaded!")
            return True
        except Exception as e:
            print(f"❌ NLLB loading failed: {e}")
            return False
    
    def load_m2m100_model(self):
        """Load M2M-100 model"""
        if "m2m100" in self.translation_models:
            return True
            
        model_name = "facebook/m2m100_418M"
        print(f"🔄 Loading M2M-100 model...")
        
        try:
            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            model = M2M100ForConditionalGeneration.from_pretrained(model_name)
            
            self.translation_models["m2m100"] = {
                "tokenizer": tokenizer,
                "model": model,
                "type": "m2m100"
            }
            
            print("✅ M2M-100 model loaded!")
            return True
        except Exception as e:
            print(f"❌ M2M-100 loading failed: {e}")
            return False
    
    def load_marian_model(self, source_lang, target_lang):
        """Load Marian model"""
        key = f"marian_{source_lang}_{target_lang}"
        if key in self.translation_models:
            return True
            
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            self.translation_models[key] = {
                "tokenizer": tokenizer,
                "model": model,
                "type": "marian"
            }
            
            return True
        except Exception as e:
            print(f"❌ Marian loading failed: {e}")
            return False
    
    def basic_translate(self, text, source_lang, target_lang, model_preference="auto"):
        """Get basic translation using specialized models"""
        
        # Try NLLB first for Korean
        if (source_lang == "ko" or target_lang == "ko") and source_lang in self.nllb_codes and target_lang in self.nllb_codes:
            if self.load_nllb_model():
                return self.translate_with_nllb(text, source_lang, target_lang)
        
        # Try M2M-100
        if self.load_m2m100_model():
            return self.translate_with_m2m100(text, source_lang, target_lang)
        
        # Fallback to Marian
        if self.load_marian_model(source_lang, target_lang):
            return self.translate_with_marian(text, source_lang, target_lang)
        
        return "Error: Could not load any translation model"
    
    def translate_with_nllb(self, text, source_lang, target_lang):
        """Translate with NLLB"""
        model_info = self.translation_models["nllb"]
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        
        src_code = self.nllb_codes.get(source_lang, source_lang)
        tgt_code = self.nllb_codes.get(target_lang, target_lang)
        
        try:
            tokenizer.src_lang = src_code
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            
            with torch.no_grad():
                # Get correct target language token
                tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_code)
                
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=1024,
                    num_beams=5,
                    early_stopping=True
                )
            
            translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return translation
            
        except Exception as e:
            print(f"⚠️ NLLB error: {e}")
            return f"NLLB Error: {e}"
    
    def translate_with_m2m100(self, text, source_lang, target_lang):
        """Translate with M2M-100"""
        model_info = self.translation_models["m2m100"]
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        
        try:
            tokenizer.src_lang = source_lang
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.get_lang_id(target_lang),
                    max_length=1024,
                    num_beams=5,
                    early_stopping=True
                )
            
            translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return translation
            
        except Exception as e:
            return f"M2M-100 Error: {e}"
    
    def translate_with_marian(self, text, source_lang, target_lang):
        """Translate with Marian"""
        key = f"marian_{source_lang}_{target_lang}"
        model_info = self.translation_models[key]
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        
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
            return f"Marian Error: {e}"
    
    def analyze_korean_text(self, text):
        """Analyze Korean text for context clues"""
        analysis = {
            "formality": "unknown",
            "has_honorifics": False,
            "text_type": "general"
        }
        
        # Check formality level
        formal_count = sum(1 for ending in self.korean_patterns["formal_endings"] if ending in text)
        informal_count = sum(1 for ending in self.korean_patterns["informal_endings"] if ending in text)
        
        if formal_count > informal_count:
            analysis["formality"] = "formal"
        elif informal_count > formal_count:
            analysis["formality"] = "informal"
        
        # Check for honorifics
        analysis["has_honorifics"] = any(hon in text for hon in self.korean_patterns["honorifics"])
        
        return analysis
    
    def refine_with_llm(self, original_text, basic_translation, source_lang, target_lang, context=""):
        """Use local LLM to refine translation"""
        if self.llm_model is None:
            print("⚠️ No LLM loaded, skipping refinement")
            return basic_translation
        
        # Get model device
        model_device = next(self.llm_model.parameters()).device
        
        # Special handling for Korean
        korean_context = ""
        if source_lang == "ko":
            analysis = self.analyze_korean_text(original_text)
            korean_context = f"""
Korean Analysis:
- Formality: {analysis['formality']}
- Has honorifics: {analysis['has_honorifics']}
Please preserve the appropriate tone and formality level in English."""

        # Create refinement prompt
        prompt = f"""You are an expert translator. Please improve this translation by making it more natural, accurate, and culturally appropriate.

Original {source_lang.upper()}: {original_text}

Basic translation: {basic_translation}

Context: {context}
{korean_context}

Please provide a refined {target_lang.upper()} translation that:
1. Maintains the exact meaning
2. Uses natural {target_lang.upper()} expressions  
3. Considers cultural context and tone
4. Fixes any grammatical errors

Refined translation:"""

        try:
            # Tokenize input and ensure it's on the same device as model
            inputs = self.llm_tokenizer.encode(prompt, return_tensors="pt", max_length=2048, truncation=True)
            inputs = inputs.to(model_device)  # FIXED: Move to model device
            
            # Generate refinement
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract refined translation
            if "Refined translation:" in response:
                refined = response.split("Refined translation:")[-1].strip()
                # Clean up the response
                refined = re.sub(r'\n.*', '', refined).strip()  # Take only first line
                return refined if refined else basic_translation
            
            return basic_translation
            
        except Exception as e:
            print(f"⚠️ LLM refinement error: {e}")
            return basic_translation
    
    def hybrid_translate(self, text, source_lang, target_lang, context="", use_llm=True):
        """Perform hybrid translation: specialized model + LLM refinement"""
        print(f"🔄 Hybrid translation: {source_lang} → {target_lang}")
        
        # Step 1: Basic translation
        print("🤖 Step 1: Specialized model translation...")
        basic_translation = self.basic_translate(text, source_lang, target_lang)
        
        if basic_translation.startswith("Error:"):
            return basic_translation
        
        print(f"📄 Basic: {basic_translation}")
        
        # Step 2: LLM refinement (if enabled)
        if use_llm and self.llm_model is not None:
            print("🧠 Step 2: LLM refinement...")
            refined_translation = self.refine_with_llm(
                text, basic_translation, source_lang, target_lang, context
            )
            print(f"✨ Refined: {refined_translation}")
            
            return {
                "basic": basic_translation,
                "refined": refined_translation,
                "recommended": refined_translation,
                "method": "hybrid"
            }
        else:
            return {
                "basic": basic_translation,
                "refined": basic_translation,
                "recommended": basic_translation,
                "method": "basic_only"
            }
    
    def translate_long_text(self, text, source_lang, target_lang, context="", use_llm=True):
        """Translate long text with hybrid approach"""
        chunk_size = 400
        
        if len(text) <= chunk_size:
            result = self.hybrid_translate(text, source_lang, target_lang, context, use_llm)
            return result["recommended"] if isinstance(result, dict) else result
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        translated_paragraphs = []
        
        total_paragraphs = len([p for p in paragraphs if p.strip()])
        current_paragraph = 0
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                translated_paragraphs.append("")
                continue
            
            current_paragraph += 1
            print(f"\n📄 Paragraph {current_paragraph}/{total_paragraphs}")
            
            # Handle long paragraphs
            if len(paragraph) > chunk_size:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
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
                
                # Translate chunks
                translated_chunks = []
                for i, chunk in enumerate(chunks):
                    print(f"  📝 Chunk {i+1}/{len(chunks)}")
                    result = self.hybrid_translate(chunk, source_lang, target_lang, context, use_llm)
                    chunk_translation = result["recommended"] if isinstance(result, dict) else result
                    translated_chunks.append(chunk_translation)
                
                translated_paragraphs.append(" ".join(translated_chunks))
            else:
                # Translate whole paragraph
                result = self.hybrid_translate(paragraph, source_lang, target_lang, context, use_llm)
                paragraph_translation = result["recommended"] if isinstance(result, dict) else result
                translated_paragraphs.append(paragraph_translation)
        
        return "\n\n".join(translated_paragraphs)
    
    # Document reading methods (same as before)
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
    
    def save_translated_document(self, translated_text, original_file_path, source_lang, target_lang, method="hybrid"):
        """Save translated text to a new file"""
        original_path = Path(original_file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{original_path.stem}_{source_lang}_to_{target_lang}_{method}_{timestamp}.txt"
        new_path = original_path.parent / new_name
        
        try:
            with open(new_path, 'w', encoding='utf-8') as file:
                header = f"# Advanced Hybrid Translation\n"
                header += f"# Original: {original_path.name}\n"
                header += f"# Languages: {source_lang} → {target_lang}\n"
                header += f"# Method: {method}\n"
                header += f"# LLM: {self.current_llm or 'None'}\n"
                header += f"# Translated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                header += f"# Length: {len(translated_text):,} characters\n"
                header += "=" * 60 + "\n\n"
                
                file.write(header + translated_text)
            
            print(f"💾 Saved: {new_path}")
            return str(new_path)
        except Exception as e:
            print(f"❌ Save error: {e}")
            return None
    
    def translate_document(self, file_path, source_lang, target_lang, context="", use_llm=True):
        """Complete hybrid document translation workflow"""
        print(f"\n🚀 Advanced Hybrid Document Translation")
        print(f"📄 File: {Path(file_path).name}")
        print(f"🔄 {source_lang.upper()} → {target_lang.upper()}")
        print(f"🤖 LLM: {self.current_llm or 'Not loaded'}")
        print(f"🧠 Hybrid mode: {'ON' if use_llm and self.llm_model else 'OFF'}")
        
        # Read document
        text = self.read_document(file_path)
        
        if text.startswith("Error:"):
            return text
        
        if not text.strip():
            return "Error: Document appears to be empty"
        
        print(f"📊 Length: {len(text):,} characters")
        
        # Translate
        print(f"\n🌐 Starting translation...")
        translated_text = self.translate_long_text(text, source_lang, target_lang, context, use_llm)
        
        if translated_text.startswith("Error:"):
            return translated_text
        
        # Save
        method = "hybrid" if (use_llm and self.llm_model) else "basic"
        saved_path = self.save_translated_document(translated_text, file_path, source_lang, target_lang, method)
        
        if saved_path:
            # Preview
            preview = translated_text[:300] + "..." if len(translated_text) > 300 else translated_text
            print(f"\n📋 Preview:")
            print("─" * 50)
            print(preview)
            print("─" * 50)
            
            return f"✅ Translation completed!\n📁 Saved: {saved_path}\n📊 {len(translated_text):,} characters translated"
        else:
            return "⚠️ Translation completed but failed to save"
    
    def cleanup_memory(self):
        """Clean up GPU/RAM memory"""
        if self.llm_model is not None:
            del self.llm_model
            del self.llm_tokenizer
            self.llm_model = None
            self.llm_tokenizer = None
            self.current_llm = None
        
        # Clear translation models if needed
        # self.translation_models.clear()
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        print("🧹 Memory cleaned up")

def main():
    """Main interface for advanced hybrid translation"""
    translator = AdvancedHybridTranslator()
    
    print("🚀 Advanced Hybrid Translation System")
    print("=" * 60)
    print("🎯 Optimized for Korean ↔ English and other challenging pairs")
    print("🤖 Combines specialized models + Local LLM refinement")
    print("📄 Supports: .txt, .pdf, .docx documents")
    print("=" * 60)
    
    while True:
        print("\n" + "─" * 60)
        print("🎯 MAIN OPTIONS:")
        print("1. 🤖 Load LLM Model (for hybrid mode)")
        print("2. 📄 Translate Document (MAIN FEATURE)")
        print("3. 📝 Quick hybrid translation")
        print("4. 🔍 Compare translation methods")
        print("5. 🌐 Show available LLM models")
        print("6. 🧹 Cleanup memory")
        print("7. 🚪 Exit")
        
        choice = input(f"\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            # Load LLM
            print("\n🤖 LOAD LLM MODEL")
            print("─" * 30)
            translator.show_available_llms()
            
            print("Recommended for Korean translation:")
            print("- 'qwen-chat' (good balance of quality/speed)")
            print("- 'qwen-large' (best quality, needs more RAM)")
            print("- 'deepseek-llm' (good general purpose)")
            
            model_choice = input("\nEnter model name (e.g., 'qwen-chat'): ").strip().lower()
            
            if model_choice in translator.available_llms:
                success = translator.load_llm_model(model_choice)
                if success:
                    print(f"✅ {model_choice} ready for hybrid translation!")
                else:
                    print("❌ Failed to load model. Try a smaller one or check RAM.")
            else:
                print("❌ Invalid model choice")
        
        elif choice == "2":
            # Document translation
            print("\n📄 HYBRID DOCUMENT TRANSLATION")
            print("─" * 40)
            
            file_path = input("Enter document path: ").strip().strip('"')
            
            if not file_path:
                print("❌ Please provide a file path")
                continue
            
            print("\nLanguage codes:")
            print("Korean: ko, English: en, Japanese: ja, Chinese: zh")
            print("Spanish: es, French: fr, German: de, etc.")
            
            source_lang = input("Source language: ").strip().lower()
            target_lang = input("Target language: ").strip().lower()
            context = input("Context (optional): ").strip()
            
            use_llm = True
            if translator.llm_model is None:
                print("\n⚠️ No LLM loaded. Translation will use specialized models only.")
                use_llm_input = input("Continue? (y/n): ").strip().lower()
                if use_llm_input != 'y':
                    continue
                use_llm = False
            
            if source_lang and target_lang:
                result = translator.translate_document(file_path, source_lang, target_lang, context, use_llm)
                print(f"\n{result}")
            else:
                print("❌ Please provide both languages")
        
        elif choice == "3":
            # Quick translation
            print("\n📝 QUICK HYBRID TRANSLATION")
            print("─" * 35)
            
            text = input("Enter text: ").strip()
            source_lang = input("Source language: ").strip().lower()
            target_lang = input("Target language: ").strip().lower()
            context = input("Context (optional): ").strip()
            
            use_llm = translator.llm_model is not None
            
            if text and source_lang and target_lang:
                result = translator.hybrid_translate(text, source_lang, target_lang, context, use_llm)
                
                print(f"\n📝 Original: {text}")
                if isinstance(result, dict):
                    print(f"🤖 Basic: {result['basic']}")
                    print(f"✨ Refined: {result['refined']}")
                    print(f"📋 Method: {result['method']}")
                else:
                    print(f"🌐 Translation: {result}")
        
        elif choice == "4":
            # Compare methods
            print("\n🔍 COMPARE TRANSLATION METHODS")
            print("─" * 40)
            
            text = input("Enter text to compare: ").strip()
            source_lang = input("Source language: ").strip().lower()
            target_lang = input("Target language: ").strip().lower()
            
            if text and source_lang and target_lang:
                print(f"\n🔍 Comparing methods for: '{text}'")
                print("=" * 50)
                
                # Basic translation
                basic = translator.basic_translate(text, source_lang, target_lang)
                print(f"🤖 Specialized model only: {basic}")
                
                # Hybrid translation
                if translator.llm_model:
                    hybrid = translator.hybrid_translate(text, source_lang, target_lang, "", True)
                    if isinstance(hybrid, dict):
                        print(f"✨ Hybrid (refined): {hybrid['refined']}")
                    else:
                        print(f"✨ Hybrid: {hybrid}")
                else:
                    print("⚠️ Load LLM first to see hybrid results")
        
        elif choice == "5":
            translator.show_available_llms()
            
        elif choice == "6":
            translator.cleanup_memory()
            
        elif choice == "7":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()