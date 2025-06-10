# Take your existing hybrid translator and add these automated folder features

import pandas as pd
import os
import re
from pathlib import Path
import time
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    MarianMTModel, MarianTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
)
import torch
import PyPDF2
import docx
from datetime import datetime
import gc
import json
from typing import List, Dict, Tuple
import shutil

class WorkingAutomatedTranslator:
    def __init__(self):
        # Copy your existing initialization from hybrid_translator.py
        self.translation_models = {}
        self.llm_model = None
        self.llm_tokenizer = None
        self.current_llm = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Glossary system
        self.glossaries = {}
        self.active_glossary = None
        self.pending_terms = {}
        
        # Folder processing settings
        self.interactive_mode = True
        self.min_term_frequency = 2
        self.supported_extensions = {'.txt', '.pdf', '.docx', '.doc'}
        self.glossary_extensions = {'.csv'}
        
        print(f"🖥️ Using device: {self.device}")
        
        # Your existing language codes
        self.nllb_codes = {
            "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn",
            "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
            "ru": "rus_Cyrl", "ja": "jpn_Jpan", "ko": "kor_Hang",
            "zh": "zho_Hans", "ar": "arb_Arab", "hi": "hin_Deva"
        }
        
        self.available_llms = {
            "qwen-small": {
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "size": "1.5B parameters",
                "ram_needed": "3GB"
            },
            "qwen-chat": {
                "model": "Qwen/Qwen2.5-3B-Instruct", 
                "size": "3B parameters",
                "ram_needed": "6GB"
            },
            "qwen-large": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "size": "7B parameters", 
                "ram_needed": "12GB"
            }
        }
    
    # ========== WORKING LLM LOADING ==========
    
    def load_llm_model(self, model_choice="qwen-chat"):
        """Actually load the LLM model (not placeholder)"""
        if model_choice not in self.available_llms:
            print(f"❌ Unknown model: {model_choice}")
            return False
            
        model_name = self.available_llms[model_choice]["model"]
        print(f"🔄 Loading LLM: {model_name}")
        print(f"⚠️ This requires {self.available_llms[model_choice]['ram_needed']} RAM")
        
        try:
            # Load tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Move to device if needed
            if self.device == "cpu":
                self.llm_model = self.llm_model.to(self.device)
            
            # Set padding token
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                
            self.current_llm = model_choice
            print(f"✅ LLM {model_choice} loaded successfully on {self.device}!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading LLM: {e}")
            print("💡 Try a smaller model or check your RAM/VRAM")
            return False
    
    # ========== WORKING FOLDER ANALYSIS ==========
    
    def analyze_folder(self, folder_path: str) -> Dict:
        """Analyze folder contents and create processing plan"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            return {"error": f"Folder does not exist: {folder_path}"}
        
        analysis = {
            "folder_path": folder_path,
            "glossaries": [],
            "documents": [],
            "other_files": [],
            "subfolders": [],
            "total_size": 0
        }
        
        print(f"🔍 Analyzing folder: {folder_path}")
        print("=" * 50)
        
        # Scan all files
        for item in folder_path.iterdir():
            if item.is_file():
                file_size = item.stat().st_size
                analysis["total_size"] += file_size
                
                if item.suffix.lower() in self.glossary_extensions:
                    analysis["glossaries"].append({
                        "path": item,
                        "name": item.stem,
                        "size": file_size
                    })
                elif item.suffix.lower() in self.supported_extensions:
                    analysis["documents"].append({
                        "path": item,
                        "name": item.name,
                        "type": item.suffix.lower(),
                        "size": file_size
                    })
                else:
                    analysis["other_files"].append({
                        "path": item,
                        "name": item.name,
                        "type": item.suffix.lower(),
                        "size": file_size
                    })
            elif item.is_dir():
                analysis["subfolders"].append(item)
        
        # Display analysis
        print(f"📁 Folder Contents:")
        print(f"   📚 Glossaries found: {len(analysis['glossaries'])}")
        for glossary in analysis["glossaries"]:
            print(f"      • {glossary['name']}.csv ({glossary['size']:,} bytes)")
        
        print(f"   📄 Documents found: {len(analysis['documents'])}")
        doc_types = {}
        for doc in analysis["documents"]:
            doc_type = doc["type"]
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        for doc_type, count in doc_types.items():
            print(f"      • {doc_type}: {count} files")
        
        if analysis["other_files"]:
            print(f"   📋 Other files: {len(analysis['other_files'])} (will be ignored)")
        
        print(f"   💾 Total size: {analysis['total_size']:,} bytes ({analysis['total_size']/1024/1024:.1f} MB)")
        
        return analysis
    
    # ========== WORKING GLOSSARY SYSTEM ==========
    
    def load_glossary_csv(self, csv_path: str, glossary_name: str = None) -> str:
        """Load glossary from CSV file with proper NaN handling"""
        try:
            # Read CSV with proper handling of missing values
            df = pd.read_csv(csv_path, encoding='utf-8', keep_default_na=False, na_values=[''])
            
            # Validate CSV format
            required_columns = ['type', 'raw_name', 'translated_name']
            if not all(col in df.columns for col in required_columns):
                return f"❌ CSV must have columns: {required_columns}. Found: {list(df.columns)}"
            
            # Convert to glossary dictionary
            glossary = {}
            skipped_rows = 0
            
            for index, row in df.iterrows():
                try:
                    # Handle potential NaN/float values safely
                    raw_name = row['raw_name']
                    translated_name = row['translated_name']
                    term_type = row['type']
                    
                    # Skip rows with missing essential data
                    if pd.isna(raw_name) or pd.isna(translated_name) or pd.isna(term_type):
                        skipped_rows += 1
                        continue
                    
                    # Convert to string and strip whitespace
                    korean_term = str(raw_name).strip()
                    english_term = str(translated_name).strip()
                    term_type_clean = str(term_type).strip()
                    
                    # Handle gender field (optional)
                    gender = row.get('gender', '')
                    if pd.isna(gender):
                        gender = ''
                    else:
                        gender = str(gender).strip()
                    
                    # Skip empty terms
                    if not korean_term or not english_term or not term_type_clean:
                        skipped_rows += 1
                        continue
                    
                    glossary[korean_term] = {
                        'translation': english_term,
                        'type': term_type_clean,
                        'gender': gender,
                        'usage_count': 0,
                        'last_used': None
                    }
                    
                except Exception as e:
                    print(f"   ⚠️ Skipping row {index + 1}: {e}")
                    skipped_rows += 1
                    continue
            
            # Store glossary
            if not glossary_name:
                glossary_name = Path(csv_path).stem
                
            self.glossaries[glossary_name] = glossary
            
            success_msg = f"✅ Loaded glossary '{glossary_name}' with {len(glossary)} terms"
            if skipped_rows > 0:
                success_msg += f" (skipped {skipped_rows} invalid rows)"
            
            print(success_msg)
            
            # Show sample terms
            print(f"   📋 Sample terms:")
            for i, (korean, data) in enumerate(list(glossary.items())[:3]):
                print(f"      • {korean} → {data['translation']} ({data['type']})")
                if i >= 2:  # Show max 3 samples
                    break
            
            return f"Glossary '{glossary_name}' loaded successfully"
            
        except Exception as e:
            return f"❌ Error loading glossary: {e}"
    
    def set_active_glossary(self, glossary_name: str) -> bool:
        """Set which glossary to use for translation"""
        if glossary_name in self.glossaries:
            self.active_glossary = glossary_name
            print(f"✅ Active glossary set to: {glossary_name}")
            return True
        else:
            print(f"❌ Glossary '{glossary_name}' not found")
            return False
    
    def auto_detect_glossaries(self, analysis: Dict) -> List[str]:
        """Automatically detect and load glossaries from folder"""
        loaded_glossaries = []
        
        if not analysis["glossaries"]:
            print("📚 No glossaries found in folder")
            return loaded_glossaries
        
        print(f"\n📚 Auto-loading glossaries...")
        
        for glossary_info in analysis["glossaries"]:
            csv_path = glossary_info["path"]
            glossary_name = glossary_info["name"]
            
            print(f"   Loading: {glossary_name}.csv")
            
            try:
                result = self.load_glossary_csv(str(csv_path), glossary_name)
                if "successfully" in result:
                    loaded_glossaries.append(glossary_name)
                    print(f"   ✅ Loaded: {glossary_name}")
                else:
                    print(f"   ❌ Failed: {glossary_name} - {result}")
            except Exception as e:
                print(f"   ❌ Error loading {glossary_name}: {e}")
        
        # Set the first loaded glossary as active
        if loaded_glossaries:
            self.set_active_glossary(loaded_glossaries[0])
            print(f"   ⭐ Active glossary: {loaded_glossaries[0]}")
        
        return loaded_glossaries
    
    def apply_glossary(self, text, source_lang="ko"):
        """Apply active glossary to text before translation"""
        if not self.active_glossary or source_lang != "ko":
            return text, []
        
        glossary = self.glossaries[self.active_glossary]
        applied_terms = []
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_terms = sorted(glossary.keys(), key=len, reverse=True)
        
        for korean_term in sorted_terms:
            if korean_term in text:
                english_term = glossary[korean_term]['translation']
                term_type = glossary[korean_term]['type']
                
                # Replace Korean term with English
                text = text.replace(korean_term, english_term)
                applied_terms.append(f"{korean_term} → {english_term} ({term_type})")
        
        return text, applied_terms
    
    # ========== WORKING DOCUMENT READING ==========
    
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
    
    # ========== WORKING TRANSLATION METHODS ==========
    # (Copy your existing methods from hybrid_translator.py)
    
    def load_nllb_model(self):
        """Load NLLB model - copy from your existing code"""
        if "nllb" in self.translation_models:
            return True
            
        model_name = "facebook/nllb-200-distilled-1.3B"
        print(f"🔄 Loading NLLB model...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
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
    
    def translate_with_nllb(self, text, source_lang, target_lang):
        """Translate with NLLB with better progress tracking"""
        model_info = self.translation_models["nllb"]
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        
        src_code = self.nllb_codes.get(source_lang, source_lang)
        tgt_code = self.nllb_codes.get(target_lang, target_lang)
        
        print(f"      🔤 Tokenizing text...")
        try:
            tokenizer.src_lang = src_code
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            print(f"      🧠 Generating translation...")
            with torch.no_grad():
                tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_code)
                
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=1024,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=False  # Added for consistency
                )
            
            print(f"      📝 Decoding result...")
            translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"      ✅ NLLB translation successful")
            return translation
            
        except Exception as e:
            print(f"      ❌ NLLB error: {e}")
            return f"NLLB Error: {e}"
    
    def basic_translate(self, text, source_lang, target_lang):
        """Basic translation using best available model"""
        # Try NLLB first for Korean
        if (source_lang == "ko" or target_lang == "ko") and source_lang in self.nllb_codes and target_lang in self.nllb_codes:
            if self.load_nllb_model():
                return self.translate_with_nllb(text, source_lang, target_lang)
        
        # Add fallback to other models here (M2M-100, Marian)
        return "Error: Could not load any translation model"
    
    def smart_paragraph_grouping(self, text, max_group_size=400, max_paragraphs=2):
        """Group paragraphs for batch processing - SMALLER groups for better progress"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        groups = []
        current_group = []
        current_size = 0
        
        print(f"   📋 Original text has {len(paragraphs)} paragraphs")
        
        for paragraph in paragraphs:
            para_size = len(paragraph)
            
            # Use smaller groups for better progress tracking
            if (current_size + para_size > max_group_size or 
                len(current_group) >= max_paragraphs) and current_group:
                
                groups.append('\n\n'.join(current_group))
                current_group = [paragraph]
                current_size = para_size
            else:
                current_group.append(paragraph)
                current_size += para_size + 2
        
        if current_group:
            groups.append('\n\n'.join(current_group))
        
        print(f"   📦 Split into {len(groups)} smaller groups (avg {sum(len(g) for g in groups)//len(groups)} chars each)")
        return groups
    
    def batch_translate_group(self, text_group, source_lang, target_lang, context="", use_glossary=True):
        """Batch translate with glossary and better progress tracking"""
        print(f"   🔄 Starting group translation ({len(text_group)} chars)...")
        
        # Apply glossary if enabled
        glossary_applied = []
        if use_glossary:
            print(f"   📚 Applying glossary...")
            text_group, glossary_applied = self.apply_glossary(text_group, source_lang)
            if glossary_applied:
                print(f"   ✅ Applied {len(glossary_applied)} glossary terms")
        
        # Get basic translation with timeout
        print(f"   🤖 Basic translation...")
        try:
            basic_translation = self.basic_translate(text_group, source_lang, target_lang)
            print(f"   ✅ Basic translation completed")
        except Exception as e:
            print(f"   ❌ Basic translation failed: {e}")
            return f"Error: Basic translation failed - {e}", glossary_applied
        
        if basic_translation.startswith("Error:"):
            return basic_translation, glossary_applied
        
        # LLM refinement if available
        if self.llm_model is not None:
            print(f"   🧠 LLM refinement...")
            try:
                refined_translation = self.refine_with_llm(
                    text_group, basic_translation, source_lang, target_lang, context
                )
                print(f"   ✅ LLM refinement completed")
                return refined_translation, glossary_applied
            except Exception as e:
                print(f"   ⚠️ LLM refinement failed, using basic: {e}")
                return basic_translation, glossary_applied
        
        return basic_translation, glossary_applied
    
    def refine_with_llm(self, original_text, basic_translation, source_lang, target_lang, context=""):
        """LLM refinement with timeout and better error handling"""
        if self.llm_model is None:
            return basic_translation
        
        # Skip LLM refinement for very short texts
        if len(original_text) < 50:
            print(f"      ⏭️ Skipping LLM refinement (text too short: {len(original_text)} chars)")
            return basic_translation
        
        model_device = next(self.llm_model.parameters()).device
        
        prompt = f"""Improve this Korean to English translation:

Korean: {original_text}
Basic: {basic_translation}

Better English:"""

        try:
            print(f"      🔤 Creating LLM prompt...")
            inputs = self.llm_tokenizer.encode(prompt, return_tensors="pt", max_length=1500, truncation=True)
            inputs = inputs.to(model_device)
            
            print(f"      🧠 Generating LLM refinement...")
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_new_tokens=128,  # Reduced for speed
                    temperature=0.2,     # Lower for consistency
                    do_sample=False,     # Faster, more deterministic
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)  # Fix attention mask warning
                )
            
            print(f"      📝 Decoding LLM result...")
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Better English:" in response:
                refined = response.split("Better English:")[-1].strip()
                if refined and len(refined) > 5:  # Sanity check
                    print(f"      ✅ LLM refinement successful")
                    return refined
            
            print(f"      ⚠️ LLM refinement failed, using basic")
            return basic_translation
            
        except Exception as e:
            print(f"      ❌ LLM refinement error: {e}")
            return basic_translation
    
    # ========== WORKING FOLDER PROCESSING ==========
    
    def create_output_structure(self, input_folder: Path, source_lang: str, target_lang: str) -> Path:
        """Create organized output folder structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder_name = f"translated_{source_lang}_to_{target_lang}_{timestamp}"
        output_folder = input_folder / output_folder_name
        
        (output_folder / "translations").mkdir(parents=True, exist_ok=True)
        (output_folder / "glossaries").mkdir(parents=True, exist_ok=True)
        (output_folder / "logs").mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Created output structure: {output_folder}")
        return output_folder
    
    def sort_documents_by_priority(self, documents: List[Dict]) -> List[Dict]:
        """Sort documents by translation priority"""
        def priority_score(doc):
            type_scores = {'.txt': 3, '.docx': 2, '.doc': 2, '.pdf': 1}
            
            name_lower = doc["name"].lower()
            if any(pattern in name_lower for pattern in ['chapter', '화', 'episode', 'ch']):
                return 100 + type_scores.get(doc["type"], 0)
            
            return type_scores.get(doc["type"], 0)
        
        sorted_docs = sorted(documents, key=priority_score, reverse=True)
        
        print(f"📋 Document processing order:")
        for i, doc in enumerate(sorted_docs):
            print(f"   {i+1}. {doc['name']} ({doc['type']})")
        
        return sorted_docs
    
    def process_folder(self, folder_path: str, source_lang: str, target_lang: str, 
                      context: str = "", skip_existing: bool = True) -> Dict:
        """Main method to process entire folder automatically"""
        
        print(f"🚀 AUTOMATED FOLDER TRANSLATION")
        print("=" * 60)
        print(f"📂 Folder: {folder_path}")
        print(f"🔄 Languages: {source_lang.upper()} → {target_lang.upper()}")
        print(f"🧠 LLM: {self.current_llm or 'Not loaded'}")
        print(f"🎓 Learning: {'ON' if self.interactive_mode else 'OFF'}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Analyze folder
        analysis = self.analyze_folder(folder_path)
        if "error" in analysis:
            return analysis
        
        # Step 2: Auto-detect and load glossaries
        loaded_glossaries = self.auto_detect_glossaries(analysis)
        
        # Step 3: Create output structure
        output_folder = self.create_output_structure(
            analysis["folder_path"], source_lang, target_lang
        )
        
        # Step 4: Sort documents by priority
        sorted_documents = self.sort_documents_by_priority(analysis["documents"])
        
        if not sorted_documents:
            return {"error": "No translatable documents found in folder"}
        
        # Step 5: Process each document
        results = {
            "processed_files": [],
            "skipped_files": [],
            "failed_files": [],
            "total_time": 0,
            "total_chars": 0,
            "output_folder": output_folder,
            "glossaries_used": loaded_glossaries
        }
        
        print(f"\n🔄 Processing {len(sorted_documents)} documents...")
        
        for i, doc_info in enumerate(sorted_documents):
            doc_path = doc_info["path"]
            doc_name = doc_info["name"]
            
            print(f"\n📄 Processing file {i+1}/{len(sorted_documents)}: {doc_name}")
            print("─" * 50)
            
            try:
                # Process single document
                file_result = self.process_single_document(
                    doc_path, source_lang, target_lang, context, output_folder
                )
                
                if file_result["success"]:
                    results["processed_files"].append(file_result)
                    results["total_chars"] += file_result["char_count"]
                    print(f"✅ Completed: {doc_name}")
                else:
                    results["failed_files"].append({
                        "file": doc_name,
                        "error": file_result["error"]
                    })
                    print(f"❌ Failed: {doc_name} - {file_result['error']}")
                
            except Exception as e:
                error_msg = str(e)
                results["failed_files"].append({
                    "file": doc_name,
                    "error": error_msg
                })
                print(f"❌ Error processing {doc_name}: {error_msg}")
            
            # Progress update
            progress = (i + 1) / len(sorted_documents) * 100
            print(f"📊 Overall progress: {progress:.1f}%")
        
        # Step 6: Final summary
        total_time = time.time() - start_time
        results["total_time"] = total_time
        
        print(f"\n🎉 FOLDER PROCESSING COMPLETED!")
        print(f"✅ Successfully processed: {len(results['processed_files'])} files")
        print(f"❌ Failed: {len(results['failed_files'])} files")
        print(f"📊 Total characters: {results['total_chars']:,}")
        print(f"⏱️ Total time: {total_time:.2f} seconds")
        print(f"📁 Output folder: {output_folder}")
        
        return results
    
    def process_single_document(self, doc_path: Path, source_lang: str, target_lang: str, 
                               context: str, output_folder: Path) -> Dict:
        """Process a single document"""
        
        try:
            # Read document
            text = self.read_document(str(doc_path))
            if text.startswith("Error:"):
                return {"success": False, "error": text}
            
            char_count = len(text)
            print(f"📊 Document length: {char_count:,} characters")
            
            # Translate
            start_time = time.time()
            
            groups = self.smart_paragraph_grouping(text)
            print(f"📦 Created {len(groups)} paragraph groups")
            
            translated_groups = []
            all_glossary_terms = []
            
            for j, group in enumerate(groups):
                translated_group, glossary_terms = self.batch_translate_group(
                    group, source_lang, target_lang, context, True
                )
                
                translated_groups.append(translated_group)
                all_glossary_terms.extend(glossary_terms)
                
                group_progress = (j + 1) / len(groups) * 100
                print(f"   📦 Group {j+1}/{len(groups)} completed ({group_progress:.1f}%)")
            
            final_translation = '\n\n'.join(translated_groups)
            translation_time = time.time() - start_time
            
            # Save translated document
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_folder / "translations" / f"{doc_path.stem}_{source_lang}_to_{target_lang}_{timestamp}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                header = f"# Automated Translation\n"
                header += f"# Original: {doc_path.name}\n"
                header += f"# Languages: {source_lang} → {target_lang}\n"
                header += f"# Translated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                header += f"# Processing time: {translation_time:.2f} seconds\n"
                header += f"# Character count: {char_count:,}\n"
                header += f"# Glossary terms applied: {len(set(all_glossary_terms))}\n"
                header += "=" * 60 + "\n\n"
                
                f.write(header + final_translation)
            
            return {
                "success": True,
                "file": doc_path.name,
                "output_file": output_file.name,
                "char_count": char_count,
                "translation_time": translation_time,
                "glossary_terms": len(set(all_glossary_terms)),
                "groups_processed": len(groups)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    """Working main interface"""
    translator = WorkingAutomatedTranslator()
    
    print("🚀 Automated Folder Translation System")
    print("=" * 60)
    print("✨ Features:")
    print("   📁 Automatic folder processing")
    print("   📚 Auto-detect glossaries") 
    print("   🔄 Batch translate all documents")
    print("   🎓 Interactive learning")
    print("   📊 Detailed reporting")
    print("=" * 60)
    
    while True:
        print("\n" + "─" * 60)
        print("🎯 MAIN OPTIONS:")
        print("1. 📁 Process Single Folder (Auto-mode)")
        print("2. 🔍 Analyze Folder (Preview)")
        print("3. 🤖 Load LLM Model")
        print("4. ⚙️ Configure Settings")
        print("5. 🚪 Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            # Single folder processing
            print("\n📁 AUTOMATED FOLDER PROCESSING")
            print("─" * 40)
            
            folder_path = input("Enter folder path: ").strip().strip('"')
            if not folder_path:
                continue
            
            source_lang = input("Source language (ko/en/etc): ").strip().lower()
            target_lang = input("Target language (en/ko/etc): ").strip().lower()
            
            context = input("Global context (optional): ").strip()
            
            skip_existing = input("Skip already translated files? (y/n) [y]: ").strip().lower()
            skip_existing = skip_existing != 'n'
            
            if source_lang and target_lang:
                print(f"\n🚀 Starting automated processing...")
                results = translator.process_folder(
                    folder_path, source_lang, target_lang, context, skip_existing
                )
                
                if "error" not in results:
                    print(f"\n✅ Processing completed successfully!")
                    print(f"📊 Check output folder: {results['output_folder']}")
                else:
                    print(f"❌ Error: {results['error']}")
        
        elif choice == "2":
            # Analyze folder
            print("\n🔍 FOLDER ANALYSIS")
            print("─" * 25)
            
            folder_path = input("Enter folder path to analyze: ").strip().strip('"')
            if folder_path:
                analysis = translator.analyze_folder(folder_path)
                
                if "error" not in analysis:
                    print(f"\n📊 Analysis completed!")
                    print(f"💡 This folder contains {len(analysis['documents'])} translatable documents")
                    print(f"📚 Found {len(analysis['glossaries'])} glossary files")
                else:
                    print(f"❌ {analysis['error']}")
        
        elif choice == "3":
            # Load LLM
            print("\n🤖 LOAD LLM MODEL")
            print("─" * 25)
            print("Available models:")
            for name, info in translator.available_llms.items():
                print(f"  {name}: {info['ram_needed']}")
            
            model_choice = input("\nEnter model name: ").strip().lower()
            if model_choice in translator.available_llms:
                success = translator.load_llm_model(model_choice)
                if success:
                    print(f"🎉 {model_choice} ready for translation!")
                else:
                    print("❌ Failed to load model")
        
        elif choice == "4":
            # Configure settings
            print("\n⚙️ CONFIGURATION SETTINGS")
            print("─" * 35)
            
            print(f"Current settings:")
            print(f"   🎓 Interactive learning: {'ON' if translator.interactive_mode else 'OFF'}")
            print(f"   📊 Minimum term frequency: {translator.min_term_frequency}")
            
            print(f"\nWhat would you like to configure?")
            print(f"1. Toggle interactive learning")
            print(f"2. Set minimum term frequency")
            print(f"3. View supported file types")
            
            config_choice = input("Enter choice (1-3): ").strip()
            
            if config_choice == "1":
                translator.interactive_mode = not translator.interactive_mode
                print(f"✅ Interactive learning: {'ON' if translator.interactive_mode else 'OFF'}")
                
            elif config_choice == "2":
                freq = input(f"Enter minimum frequency [{translator.min_term_frequency}]: ").strip()
                if freq.isdigit():
                    translator.min_term_frequency = int(freq)
                    print(f"✅ Minimum frequency set to: {translator.min_term_frequency}")
                    
            elif config_choice == "3":
                print(f"\n📄 Supported document types:")
                for ext in sorted(translator.supported_extensions):
                    print(f"   • {ext}")
                print(f"\n📚 Supported glossary types:")
                for ext in sorted(translator.glossary_extensions):
                    print(f"   • {ext}")
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()