# < Premium Vietnamese Keyword Clustering

AI-powered keyword clustering using OpenAI embeddings and GPT-4o for best-in-class Vietnamese and English keyword analysis.

## ¡ Quick Start

### 1. Run the App
Double-click `run.bat` or run:
```bash
venv\Scripts\python.exe -m streamlit run app.py
```

### 2. Open Browser
Navigate to: **http://localhost:8501**

### 3. Add API Key
Place your OpenAI API key in `.env` file:
```
OPENAI_API_KEY=sk-proj-...your-key-here...
```
Get your key at: https://platform.openai.com/api-keys

### 4. Upload & Cluster
- Upload Excel file
- Select sheet and column
- Click "Start Clustering"
- Download results!

## <¯ Features

- **OpenAI text-embedding-3-large**: State-of-the-art 3072-dimensional embeddings
- **GPT-4o**: Intelligent, descriptive cluster labels
- **Vietnamese Support**: Full word segmentation and diacritics preservation
- **Zero Outliers**: Automatic outlier reassignment to nearest clusters
- **Interactive Visualizations**: 2D cluster maps, distributions, and labels

## =° Cost

Approximately **$0.50-$2.00** per 1,000 keywords
- Embeddings: ~$0.10-0.50
- GPT-4o labels: ~$1.00

## =Ê Output

Your Excel file will include:
- **keywords**: Original text (Vietnamese diacritics preserved)
- **segmented**: Word-segmented version (Vietnamese only)
- **Cluster**: AI-generated cluster ID
- **Cluster Label**: GPT-4o descriptive label

## =à Technical Stack

- **Embeddings**: OpenAI text-embedding-3-large
- **Clustering**: HDBSCAN (density-based, automatic detection)
- **Labeling**: GPT-4o
- **Vietnamese NLP**: underthesea word segmentation
- **Visualization**: Plotly + UMAP

## =Ý Languages Supported

- <û<ó **Vietnamese**: Full support with word segmentation
- <ì<ç **English**: Full support

## =' Requirements

- Python 3.12
- OpenAI API key
- Virtual environment (included in `venv/`)

## =Ú Documentation

- `SESSION_LOG.md`: Complete development log and technical details
- `CLAUDE.md`: Instructions for Claude Code

## = Issues?

If you encounter problems:
1. Ensure you're running from virtual environment
2. Check API key is in `.env` file
3. Verify internet connection for API calls
4. Check `SESSION_LOG.md` for troubleshooting

## =€ Recent Updates

**2025-11-24**:
-  Removed all free-mode code (premium-only)
-  Implemented outlier reduction (relaxed HDBSCAN + reassignment)
-  Expected outliers: 40% ’ 0%

---

**Built with Streamlit " Powered by OpenAI & GPT-4o**
