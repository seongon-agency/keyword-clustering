# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

AI-powered keyword clustering application using OpenAI embeddings and GPT-4o-mini for semantic analysis. Built with a Next.js frontend and FastAPI backend.

## Architecture

```
keyword_clustering_easy_demo/
├── backend/                 # FastAPI backend
│   ├── main.py             # API endpoints and clustering logic
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile          # Docker config for Railway
│   └── railway.toml        # Railway deployment config
├── nextjs-app/             # Next.js frontend
│   ├── src/
│   │   ├── app/           # App router pages and API routes
│   │   ├── components/    # React components
│   │   └── context/       # React context (theme)
│   ├── Dockerfile         # Docker config for Railway
│   └── railway.toml       # Railway deployment config
└── .env.example           # Environment variables template
```

## Key Technologies

- **Frontend**: Next.js 15, React 19, Tailwind CSS, Recharts, Plotly.js
- **Backend**: FastAPI, Python 3.12
- **AI/ML**: OpenAI text-embedding-3-large, GPT-4o-mini, HDBSCAN, UMAP
- **Vietnamese NLP**: underthesea word segmentation

## Clustering Pipeline

1. **Preprocessing**: Vietnamese word segmentation (underthesea)
2. **Embeddings**: OpenAI text-embedding-3-large (3,072 dimensions)
3. **Dimensionality Reduction**: UMAP (3,072 → 50 dims for clustering)
4. **Clustering**: HDBSCAN with automatic cluster detection
5. **Labeling**: GPT-4o-mini generates descriptive labels
6. **Visualization**: UMAP 2D/3D projections

## HDBSCAN Configuration

```python
min_cluster_size = max(5, n_samples // 200)  # 0.5% of dataset
min_cluster_size = min(min_cluster_size, 30) # Cap at 30
min_samples = 3
cluster_selection_method = 'eom'  # Excess of Mass
```

## Running Locally

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd nextjs-app
npm install
npm run dev
```

## Environment Variables

- `OPENAI_API_KEY`: Required for embeddings and labeling
- `BACKEND_URL`: Backend URL (defaults to http://localhost:8000)

## Deployment

Deploy to Railway using the monorepo setup:
1. Create a new project on Railway
2. Add both services (backend and frontend) from the repository
3. Set `OPENAI_API_KEY` environment variable
4. Set `BACKEND_URL` on frontend to point to backend service
