# 🚀 Advanced Hybrid Translation Tool

A powerful local machine translation system optimized for Korean ↔ English and other language pairs, featuring automated folder processing, interactive glossary learning, and hybrid AI refinement.

## ✨ Features

### 🎯 **Automated Folder Processing**
- **📁 One-click folder translation** - Point to folder, get all documents translated
- **📚 Auto-detect glossaries** - Automatically loads CSV glossaries from folder
- **🔄 Batch processing** - Translates multiple documents efficiently
- **📊 Smart prioritization** - Processes chapters and important files first
- **📁 Organized output** - Creates structured output folders with logs

### 🤖 **Hybrid AI Translation**
- **🧠 Multi-model approach** - NLLB + M2M-100 + Marian for base translation
- **✨ LLM refinement** - Qwen/DeepSeek models for natural language improvement
- **🎯 Context-aware** - Understands genre, formality, and cultural context
- **⚡ GPU acceleration** - Optimized for NVIDIA RTX cards (4060+ recommended)

### 📚 **Interactive Glossary System**
- **🔍 Auto-term detection** - AI identifies potential glossary terms
- **💡 Smart suggestions** - Frequency-based and pattern-based recommendations  
- **🎓 Interactive learning** - Review and approve new terms after translation
- **📋 OpenNovel compatible** - Works with existing CSV glossaries
- **💾 Auto-save updates** - Glossaries improve over time

### 🌐 **Language & Quality Features**
- **🇰🇷 Korean optimized** - Special handling for honorifics, formality levels
- **📄 Multiple formats** - Supports .txt, .pdf, .docx documents
- **🎭 Context analysis** - Automatically detects genre and adjusts translation style
- **🔄 Consistent terminology** - Maintains character names and terms across documents
- **📊 Quality metrics** - Detailed reporting and progress tracking

## 🔧 Installation

### Prerequisites
- **Python 3.9+** (3.11 recommended for best compatibility)
- **8GB+ RAM** (12GB+ recommended for large models)
- **NVIDIA GPU** (Optional but highly recommended - RTX 4060+ with 8GB+ VRAM)

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/advanced-hybrid-translator.git
   cd advanced-hybrid-translator
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac  
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   # For NVIDIA GPU (Recommended)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   
   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

4. **Run the translator**:
   ```bash
   python ultimateTranslator.py
   ```

## 🚀 Quick Start

### Basic Workflow

1. **Organize your project folder**:
   ```
   my_novel/
   ├── glossary.csv              # Character names, terms (optional)
   ├── chapter_001.txt           # Documents to translate
   ├── chapter_002.docx
   └── chapter_003.pdf
   ```

2. **Run the translator**:
   ```bash
   python ultimateTranslator.py
   ```

3. **Load AI model** (choose based on your hardware):
   - `qwen-small` - 3GB RAM, good quality
   - `qwen-chat` - 6GB RAM, better quality  
   - `qwen-large` - 12GB RAM, best quality

4. **Process folder**:
   - Select "📁 Process Single Folder"
   - Enter folder path: `my_novel`
   - Languages: `ko` → `en`
   - Context: `"Fantasy light novel with action scenes"`

5. **Get results**:
   ```
   my_novel/
   └── translated_ko_to_en_20250610_162440/
       ├── translations/           # Your translated files
       ├── glossaries/            # Updated glossaries  
       └── logs/                  # Processing reports
   ```

### Glossary Format

Create CSV files compatible with OpenNovel format:

```csv
type,raw_name,translated_name,gender
character,이시헌,Lee Si-heon,male
character,산수유,Sansuyu,female
location,세계수,World Tree,
organization,플라워,Flower,
skill,개화,Bloom,
title,목령왕,Necromancy King,male
```

## 💡 Usage Examples

### Novel Translation Project
```bash
# Folder structure:
novel_series/
├── character_glossary.csv      # Main characters
├── world_glossary.csv          # Places, terms, skills
├── volume_1/
│   ├── ch001.txt
│   ├── ch002.txt
│   └── ch003.txt
└── volume_2/
    ├── ch001.txt
    └── ch002.txt

# Single command translates everything:
# 1. Auto-loads both glossaries
# 2. Processes all chapters in order
# 3. Learns new terms interactively
# 4. Saves organized output
```

### Web Novel Chapter Batch
```bash
# Raw chapters from web scraping:
webnovel_chapters/
├── terms_glossary.csv
├── chapter_390.txt
├── chapter_391.txt
├── chapter_392.txt
└── chapter_393.txt

# Result: Consistent translation across all chapters
# with automatically updated glossary
```

## ⚙️ Configuration

### Translation Models

| Model | RAM Needed | Quality | Speed | Best For |
|-------|------------|---------|-------|----------|
| NLLB-200 | 2GB | Excellent | Fast | Base translation |
| M2M-100 | 1GB | Very Good | Very Fast | Fallback model |
| Qwen-Small | 3GB | Good | Fast | LLM refinement |
| Qwen-Chat | 6GB | Very Good | Medium | Balanced quality/speed |
| Qwen-Large | 12GB | Excellent | Slow | Best quality |

### Context Examples

| Genre | Good Context |
|-------|--------------|
| Fantasy | `"Fantasy light novel with magic and monsters"` |
| Romance | `"Modern romance, informal dialogue between friends"` |
| Action | `"Action manhwa with fighting scenes and power systems"` |
| Historical | `"Historical drama set in Joseon period, formal speech"` |

### Hardware Recommendations

| Hardware | Model Recommendation | Expected Performance |
|----------|---------------------|---------------------|
| RTX 4090 | Qwen-Large + NLLB | ~2000 chars/second |
| RTX 4060 | Qwen-Chat + NLLB | ~800 chars/second |
| RTX 3060 | Qwen-Small + NLLB | ~500 chars/second |
| CPU Only | M2M-100 only | ~200 chars/second |

## 🛠️ Advanced Features

### Interactive Learning Session
```
📚 GLOSSARY UPDATE SESSION
========================================
🔍 Found 3 potential terms to add:

1. 🔤 Korean: '시스투스'
   🌐 Suggested English: 'Sistus'
   📂 Type: character
   🎯 Confidence: 0.9
   Actions: [a]dd / [s]kip / [e]dit / [q]uit review
```

### Batch Processing Options
- **Skip existing** - Don't retranslate already processed files
- **Recursive mode** - Process subfolders automatically  
- **Custom contexts** - Different contexts for different folders
- **Progress tracking** - Real-time progress with ETA

### Quality Controls
- **Terminology consistency** - Maintains character/place names
- **Cultural context** - Preserves honorifics and formality levels
- **Genre awareness** - Adapts translation style to content type
- **Error recovery** - Continues processing if individual files fail

## 📊 Output Structure

```
translated_ko_to_en_20250610_162440/
├── translations/
│   ├── chapter_001_ko_to_en_20250610_162440.txt
│   ├── chapter_002_ko_to_en_20250610_162445.txt
│   └── chapter_003_ko_to_en_20250610_162450.txt
├── glossaries/
│   ├── character_glossary_updated.csv
│   └── world_glossary_updated.csv
└── logs/
    └── processing_report.txt
```

Each translated file includes:
- **Translation metadata** - Original file, languages, processing time
- **Quality metrics** - Character count, glossary terms applied
- **Model information** - Which AI models were used
- **Clean formatted text** - Properly structured paragraphs

## 🐛 Troubleshooting

### Common Issues

**"CUDA out of memory"**
```bash
# Solution: Use smaller model
Enter model name: qwen-small  # Instead of qwen-large
```

**"Error loading glossary"**
```bash
# Check CSV format - ensure no empty required fields
# Use UTF-8 encoding
# Required columns: type, raw_name, translated_name
```

**"Translation very slow"**
```bash
# Check if using CPU instead of GPU
# Reduce batch size in settings
# Use smaller LLM model
```

**"Models not downloading"**
```bash
# Check internet connection
# Ensure enough disk space (models are 1-8GB each)
# Try clearing transformers cache: ~/.cache/huggingface/
```

### Performance Optimization

1. **GPU Setup**:
   ```bash
   # Verify CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Memory Management**:
   - Close other applications
   - Use smaller models for large documents
   - Process files in smaller batches

3. **Speed Tips**:
   - Use SSD storage for model cache
   - Disable LLM refinement for draft translations
   - Use batch processing for multiple files

## 📜 License & Attribution

### MIT License

Copyright (c) 2025 Advanced Hybrid Translation Tool

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

**🏷️ Attribution Required:**

When using, modifying, or distributing this software, you must:
- ✅ **Keep the original copyright notice** in all copies
- ✅ **Credit the original author** in any derivative works  
- ✅ **Include this license text** in distributions
- ✅ **Reference the original repository** in documentation

**Example attribution:**
```
Based on Advanced Hybrid Translation Tool
Original work by [Your Name]
https://github.com/yourusername/advanced-hybrid-translator
```

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 🙏 Acknowledgments

- **Hugging Face Transformers** - For providing excellent model infrastructure
- **Facebook AI Research** - For NLLB and M2M-100 translation models  
- **Alibaba Cloud** - For Qwen language models
- **DeepSeek** - For advanced language model capabilities
- **OpenNovel community** - For glossary format inspiration
- **Korean web novel community** - For testing and feedback

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for contribution:
- 🌐 Additional language pair support
- 🔧 Performance optimizations  
- 📚 New glossary formats
- 🎨 UI improvements
- 📖 Documentation and tutorials
- 🧪 Testing on different hardware setups

## ⚠️ Disclaimer

This tool is for personal and educational use. Please respect:
- **Copyright laws** - Only translate content you have rights to use
- **Website terms of service** - Don't violate scraping restrictions  
- **Model licenses** - Follow individual model usage terms
- **Fair use principles** - Use responsibly and ethically

The authors are not responsible for misuse of this tool or any legal issues arising from its use.

---

**Made with ❤️ for the global translation community**

*Bringing advanced AI translation to everyone, everywhere.*