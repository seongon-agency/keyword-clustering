# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains two keyword clustering tools for SEO analysis:

1. **`cl.py`**: Original CLI script optimized for Italian text with FastText, Word2Vec, and BERTopic clustering
2. **`app.py`**: Streamlit web application optimized for English and Vietnamese with Word2Vec and BERTopic clustering (recommended for most users)

Both perform advanced NLP-based clustering using multiple techniques, but the Streamlit app provides a modern web interface and better language support for English/Vietnamese.

## Streamlit Web App (app.py) - Recommended

### Running the Streamlit App

**Prerequisites**:
```bash
pip install -r requirements.txt
```

**Launch**:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Features

- **Web-based UI**: Upload Excel files, select columns, and download results through a modern interface
- **Language Support**: Optimized for English and Vietnamese (automatic stopword detection)
- **Progress Tracking**: Real-time status updates and progress bars for each processing stage
- **Model Selection**: Choose between multiple embedding models via sidebar
- **Adjustable Parameters**: Configure clustering blocks (100-2000) via slider
- **Interactive Visualizations**: Comprehensive cluster visualizations with UMAP, Plotly, and BERTopic built-in charts

### Architecture

The Streamlit app follows a simplified pipeline:
1. **File Upload**: Direct file upload via web interface (no folder structure needed)
2. **Text Preprocessing**: Language-aware cleaning with English/Vietnamese stopwords
3. **Stemming**: Snowball stemmer with frequency-based ranking
4. **Word2Vec**: Custom-trained embeddings on the dataset
5. **Word2Vec Clustering**: HDBSCAN clustering in configurable blocks
6. **BERTopic**: Multilingual transformer-based semantic clustering
7. **Export**: Download button for timestamped Excel results

### Key Differences from CLI Version

- **No FastText**: Removed to simplify dependencies and reduce model download size
- **No Lemmatization**: Removed spaCy dependency for lighter installation
- **Multilingual Models**: Uses `paraphrase-multilingual-MiniLM-L12-v2` by default
- **Cached Models**: Streamlit caching prevents reloading models between runs
- **No Folder Structure**: Upload files directly, no `input/output` folders needed

### Embedding Models Available

Selectable via sidebar dropdown:
- `paraphrase-multilingual-MiniLM-L12-v2` (default, recommended)
- `paraphrase-multilingual-mpnet-base-v2` (higher quality, slower)
- `multilingual-e5-base` (good balance)

### Output Columns

- `keywords`: Original keyword text
- `Cleaned`: Preprocessed text (lowercase, no punctuation, no stopwords)
- `Stemmi`: Stemmed version
- `top stems`, `2 top stems`, `3 top stems`: Top N most frequent stems
- `w2v`: Word2Vec embedding vectors (300-dim)
- `W2V Cluster`: Word2Vec HDBSCAN cluster ID
- `Bertopic`: Semantic topic label (descriptive text)

### Interactive Visualizations

The app provides four visualization tabs after clustering completes:

**1. Word2Vec Scatter Plot**
- 2D scatter plot of all keywords using UMAP dimensionality reduction
- Interactive hover to see individual keywords
- Color-coded by cluster ID
- Helps identify cluster separation and outliers

**2. Cluster Distribution**
- Side-by-side bar charts for Word2Vec and BERTopic clusters
- Shows number of keywords per cluster
- Color-coded by size
- Helps identify dominant vs. minor clusters

**3. Top Keywords per Cluster**
- Sample keywords from each cluster (10 for W2V, 5 for BERTopic)
- Quick reference to understand cluster content
- Side-by-side comparison of both clustering methods

**4. BERTopic Advanced Visualizations**
- **Top Words per Topic**: Bar chart showing most important words per topic
- **Intertopic Distance Map**: Interactive 2D map showing topic relationships in semantic space
- **Topic Hierarchy**: Dendrogram showing hierarchical structure of topics

All visualizations are interactive (zoom, pan, hover) using Plotly and update automatically after each clustering run.

### Technical Details - Visualizations

**UMAP Configuration** (app.py:248-260):
- `n_neighbors=15`: Controls local vs. global structure balance
- `min_dist=0.1`: Minimum distance between points in low-dimensional space
- `metric='cosine'`: Distance metric (appropriate for text embeddings)
- `n_components=2`: Reduce to 2D for visualization
- `random_state=42`: Reproducible results

**Visualization Libraries**:
- Plotly Express for scatter plots
- Plotly Graph Objects for custom bar charts
- BERTopic built-in visualizations for topic analysis
- UMAP for dimensionality reduction

## Original CLI Script (cl.py) - Italian Language

### Core Architecture

The script follows a linear pipeline architecture with these main stages:

1. **Data Import** (lines 53-126): Interactive Excel file/sheet/column selection from `input/` folder
2. **Text Preprocessing** (lines 130-166): Cleaning, stopword removal, deduplication
3. **Stemming** (lines 171-232): Snowball stemmer for Italian, with frequency-based top stem extraction
4. **Lemmatization** (lines 237-299): spaCy Italian model for lemmatization, with frequency-based top lemma extraction
5. **FastText Embeddings** (lines 302-334): Pre-trained Italian FastText model (`cc.it.300.bin`)
6. **FastText Clustering** (lines 337-375): HDBSCAN clustering in blocks (1000 blocks by default)
7. **Word2Vec** (lines 377-408): Custom-trained Word2Vec on the dataset
8. **Word2Vec Clustering** (lines 412-451): HDBSCAN clustering in blocks
9. **BERTopic Clustering** (lines 454-500): Semantic topic modeling with labeled clusters
10. **Export** (lines 503-519): Timestamped Excel output to `output/` folder

### Key Design Patterns

**Block Processing**: Large datasets are processed in blocks (default n_blocks=1000) for both FastText and Word2Vec clustering to manage memory efficiently. This is critical for handling large keyword lists.

**Frequency-Based Ranking**: Both stems and lemmas are ranked by global frequency, then the top N most frequent forms are selected per keyword. This creates the "top stems", "2 top stems", "3 top stems" and equivalent lemma columns.

**Multiple Embedding Methods**: The CLI script generates three different clustering approaches simultaneously:
- FastText (column: "FT Cluster"): Pre-trained embeddings
- Word2Vec (column: "W2V Cluster"): Dataset-specific embeddings
- BERTopic (column: "Bertopic"): Semantic topic labels (text descriptions, not numeric IDs)

### Running the CLI Script

**Prerequisites**:
```bash
pip install pandas numpy sentence-transformers bertopic openpyxl spacy nltk fasttext hdbscan gensim xlsxwriter
python -m spacy download it_core_news_lg
```

**Execution**:
```bash
python cl.py
```

The script is interactive and will prompt for:
1. Which Excel file from `input/` folder
2. Which sheet from the file
3. Which column contains the keywords

**Output**: Results are saved to `output/cluster_YYYYMMDD_HHMM.xlsx` with all original data plus clustering columns.

### Important Configuration Points

**BERTopic Model Selection** (lines 462-473): Several pre-configured embedding models are available. The active model is:
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

Commented alternatives include Italian-specific models like:
- `dbmdz/bert-base-italian-uncased`
- `nickprock/sentence-bert-base-italian-uncased`
- `Musixmatch/umberto-commoncrawl-cased-v1`

To switch models, uncomment the desired model and comment out the current one at line 472.

**Clustering Block Size** (lines 340, 415): Set via `n_blocks` variable. Increase for lower memory usage, decrease for potentially better clustering quality on smaller datasets.

**HDBSCAN Parameters** (lines 359, 435):
- `min_cluster_size=2`: Minimum points to form a cluster
- `min_samples=1`: Core point neighborhood size
Adjust these for stricter/looser clustering.

### DataFrame Column Structure

Final output columns:
- `keywords`: Original keyword text
- `Cleaned`: Lowercase, no punctuation, no stopwords, no duplicates
- `Stemmi`: Stemmed version of cleaned text
- `top stems`, `2 top stems`, `3 top stems`: Top N most frequent stems
- `Lemmi`: Lemmatized version of cleaned text
- `top lemma`, `2 top lemmas`, `3 top lemmas`: Top N most frequent lemmas
- `emb ft`: FastText embedding vectors (300-dim)
- `FT Cluster`: FastText HDBSCAN cluster ID
- `w2v`: Word2Vec embedding vectors (300-dim)
- `W2V Cluster`: Word2Vec HDBSCAN cluster ID
- `Bertopic`: BERTopic cluster label (descriptive text)

### Critical Implementation Details

**Italian Language Models**: The script uses `it_core_news_lg` for spaCy lemmatization and downloads Italian FastText model on first run. Ensure these are installed.

**Folder Structure**: Script auto-creates `input/` and `output/` folders in the script directory if they don't exist (lines 40-49).

**Memory Management**: Block-based processing is essential for large datasets (>10k keywords). The script divides embeddings into blocks and clusters each independently, which may result in different cluster IDs for similar keywords across blocks.

**BERTopic Labels**: Unlike numeric cluster IDs, BERTopic generates human-readable topic descriptions by joining the top words for each topic (line 488). The -1 topic represents outliers and is labeled "No Topic".
