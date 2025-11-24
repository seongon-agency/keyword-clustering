import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import re
from collections import Counter
import unicodedata

# NLP and Clustering libraries
import hdbscan

# Premium features - OpenAI & Vietnamese NLP
from openai import OpenAI
from underthesea import word_tokenize
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Visualization libraries
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP

# Page configuration
st.set_page_config(
    page_title="Keyword Clustering Tool",
    page_icon="🔍",
    layout="wide"
)

st.title("🌟 Premium Vietnamese Keyword Clustering")
st.markdown("**AI-Powered Clustering with OpenAI & GPT-4o** - Best-in-class quality for Vietnamese keywords")

# Language selection
language = st.sidebar.selectbox(
    "Select Language",
    ["Vietnamese", "English"],
    help="Choose the language of your keywords"
)

# OpenAI API Key Configuration
st.sidebar.subheader("🔑 API Configuration")

# Load API key from environment variable
env_api_key = os.getenv("OPENAI_API_KEY")

if env_api_key:
    openai_api_key = env_api_key
    st.sidebar.success("✓ API Key loaded from .env file")
else:
    # Fallback to manual input if .env not found
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    if openai_api_key:
        st.sidebar.success("✓ API Key configured")
    else:
        st.sidebar.error("⚠️ OpenAI API key required! Add it to .env file or enter above.")

# Cost estimate
if openai_api_key:
    st.sidebar.info("💰 Estimated cost: $0.50-$2.00 per 1000 keywords")

# Advanced Settings
st.sidebar.subheader("⚙️ Advanced Settings")
clustering_blocks = st.sidebar.slider(
    "Clustering Blocks",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100,
    help="Number of blocks for processing. Increase for lower memory usage."
)

def get_openai_embeddings(keywords, api_key, model="text-embedding-3-large"):
    """Get embeddings from OpenAI API"""
    with st.status("Getting OpenAI embeddings...", expanded=True) as status:
        try:
            client = OpenAI(api_key=api_key)
            st.write(f"Processing {len(keywords)} keywords with {model}...")

            # OpenAI has a limit of 8191 tokens per request, process in batches
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(keywords), batch_size):
                batch = keywords[i:i + batch_size]
                st.write(f"Processing batch {i // batch_size + 1}/{(len(keywords) + batch_size - 1) // batch_size}...")

                response = client.embeddings.create(
                    model=model,
                    input=batch,
                    encoding_format="float"
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            status.update(label=f"OpenAI embeddings complete! ({len(all_embeddings)} embeddings)", state="complete")
            return np.array(all_embeddings)
        except Exception as e:
            st.error(f"Error getting OpenAI embeddings: {str(e)}")
            st.info("Falling back to free model...")
            return None

def generate_cluster_labels_with_gpt4(df, cluster_column, api_key, language="Vietnamese"):
    """Generate high-quality cluster labels using GPT-4o"""
    with st.status("Generating cluster labels with GPT-4o...", expanded=True) as status:
        try:
            client = OpenAI(api_key=api_key)
            unique_clusters = df[cluster_column].unique()
            cluster_labels = {}

            for i, cluster_id in enumerate(unique_clusters):
                if cluster_id == -1:
                    cluster_labels[cluster_id] = "Outliers / Uncategorized"
                    continue

                # Get sample keywords from this cluster
                cluster_keywords = df[df[cluster_column] == cluster_id]['keywords'].head(15).tolist()
                keywords_str = "\n".join(f"- {kw}" for kw in cluster_keywords)

                st.write(f"Processing cluster {i + 1}/{len(unique_clusters)}...")

                prompt = f"""Analyze these {language} keywords and create a concise, descriptive label for this semantic cluster.

Keywords in this cluster:
{keywords_str}

Requirements:
1. Provide a short label (2-5 words) in {language} that captures the main theme
2. Be specific and descriptive
3. Use the most representative keywords
4. Respond with ONLY the label, no explanation

Label:"""

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=50
                )

                label = response.choices[0].message.content.strip()
                cluster_labels[cluster_id] = label

            status.update(label="GPT-4o cluster labels generated!", state="complete")
            return cluster_labels
        except Exception as e:
            st.error(f"Error generating GPT-4o labels: {str(e)}")
            return None

# Core Functions
def apply_vietnamese_word_segmentation(keywords, language):
    """Apply Vietnamese word segmentation using underthesea"""
    if language != "Vietnamese":
        return keywords  # No segmentation for non-Vietnamese

    segmented = []
    with st.status("Applying Vietnamese word segmentation...", expanded=True) as status:
        st.write(f"Processing {len(keywords)} keywords...")
        for i, keyword in enumerate(keywords):
            try:
                # Tokenize using underthesea (handles multi-word tokens)
                tokens = word_tokenize(keyword)
                segmented_keyword = " ".join(tokens)
                segmented.append(segmented_keyword)
            except:
                # Fallback to original if segmentation fails
                segmented.append(keyword)

            if i % 100 == 0 and i > 0:
                st.write(f"Processed {i}/{len(keywords)} keywords...")

        status.update(label="Word segmentation complete!", state="complete")
    return segmented

def clean_keywords(df, stop_words):
    """Minimal cleaning for Vietnamese: preserve diacritics, remove duplicates only"""
    with st.status("Processing keywords...", expanded=True) as status:
        original_row_count = len(df)
        st.write(f"Original rows: {original_row_count}")

        # Only basic cleaning: lowercase and normalize whitespace
        # PRESERVE Vietnamese diacritics (no Unicode normalization)
        df["keywords"] = df["keywords"].str.strip()
        df["keywords"] = df["keywords"].str.replace(r"\s+", " ", regex=True)

        # Remove exact duplicates only (case-insensitive)
        df = df.drop_duplicates(subset=["keywords"], keep="first").reset_index(drop=True)
        df = df[df["keywords"].str.strip() != ""]

        final_row_count = len(df)
        st.write(f"Remaining rows: {final_row_count}")
        st.write(f"Removed duplicates: {original_row_count - final_row_count}")
        status.update(label="Keywords processed!", state="complete")

    return df

def apply_stemming(df, language):
    """Apply stemming based on language"""
    with st.status("Applying stemming...", expanded=True) as status:
        stemmer_lang = "english" if language == "English" else "english"  # Vietnamese not available, fallback
        stemmer = SnowballStemmer(stemmer_lang)
        cleaned_terms = df["Cleaned"].tolist()
        stemmed_terms = []

        progress_bar = st.progress(0)
        for idx, term in enumerate(cleaned_terms):
            stemmed_terms.append(" ".join(stemmer.stem(word) for word in term.split()))
            if idx % 100 == 0:
                progress_bar.progress((idx + 1) / len(cleaned_terms))

        progress_bar.progress(1.0)
        df["Stemmi"] = stemmed_terms
        status.update(label="Stemming complete!", state="complete")

    return df

def add_top_stems(df):
    """Add top stems columns based on frequency"""
    with st.status("Calculating top stems...", expanded=True) as status:
        all_stems = " ".join(df["Stemmi"]).split()
        stem_frequencies = Counter(all_stems)
        sorted_stems = [stem for stem, _ in stem_frequencies.most_common()]

        top_3_stems = []
        top_2_stems = []
        top_1_stem = []

        for stem_string in df["Stemmi"]:
            stems = stem_string.split()
            top_3_stems.append(" ".join([stem for stem in sorted_stems if stem in stems][:3]))
            top_2_stems.append(" ".join([stem for stem in sorted_stems if stem in stems][:2]))
            top_1_stem.append(" ".join([stem for stem in sorted_stems if stem in stems][:1]))

        df["3 top stems"] = top_3_stems
        df["2 top stems"] = top_2_stems
        df["top stems"] = top_1_stem

        status.update(label="Top stems calculated!", state="complete")

    return df

def train_word2vec(df):
    """Train Word2Vec model and calculate embeddings"""
    with st.status("Training Word2Vec model...", expanded=True) as status:
        # Use original keywords (preserves Vietnamese diacritics)
        tokenized_sentences = [text.split() for text in df["keywords"]]

        st.write("Training model...")
        w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=300, window=5, min_count=1, workers=4)

        st.write("Calculating embeddings...")
        embeddings = []
        for tokens in tokenized_sentences:
            word_vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
            if word_vectors:
                avg_vector = np.mean(word_vectors, axis=0)
            else:
                avg_vector = np.zeros(w2v_model.vector_size)
            embeddings.append(avg_vector)

        df["w2v"] = embeddings
        status.update(label="Word2Vec complete!", state="complete")

    return df, w2v_model

def cluster_with_hdbscan(df, embedding_column, output_column, n_blocks):
    """Cluster embeddings using HDBSCAN in blocks"""
    with st.status(f"Clustering {output_column}...", expanded=True) as status:
        embeddings = np.array(df[embedding_column].tolist())

        # For small datasets, don't use blocking
        min_block_size = 10  # HDBSCAN needs at least this many samples
        if len(embeddings) < min_block_size * 2:
            # Just cluster everything at once
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, core_dist_n_jobs=-1)
            all_clusters = clusterer.fit_predict(embeddings)
            df[output_column] = all_clusters
            status.update(label=f"{output_column} clustering complete!", state="complete")
            return df

        # Ensure n_blocks doesn't create blocks smaller than min_block_size
        max_blocks = len(embeddings) // min_block_size
        n_blocks = min(n_blocks, max_blocks)

        block_size = len(embeddings) // n_blocks
        all_clusters = np.full(len(embeddings), -1, dtype=int)

        progress_bar = st.progress(0)
        for i in range(n_blocks):
            start = i * block_size
            end = (i + 1) * block_size if i < n_blocks - 1 else len(embeddings)
            block = embeddings[start:end]

            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, core_dist_n_jobs=-1)
            block_clusters = clusterer.fit_predict(block)
            all_clusters[start:end] = block_clusters

            progress_bar.progress((i + 1) / n_blocks)

        df[output_column] = all_clusters
        status.update(label=f"{output_column} clustering complete!", state="complete")

    return df

def calculate_bertopic_clusters(df, embedding_model, language):
    """Calculate BERTopic clusters with labels"""
    with st.status("Calculating BERTopic clusters...", expanded=True) as status:
        st.write("Generating embeddings...")
        # Use original keywords (preserves Vietnamese diacritics)
        embeddings = embedding_model.encode(df["keywords"].tolist(), show_progress_bar=False)

        st.write("Running BERTopic...")
        lang = "english" if language == "English" else "english"  # BERTopic language param
        topic_model = BERTopic(language=lang, verbose=False)

        topics, _ = topic_model.fit_transform(df["keywords"].tolist(), embeddings)

        st.write("Generating topic labels...")
        topic_labels = {}
        for topic in set(topics):
            if topic != -1:
                topic_words = topic_model.get_topic(topic)
                if topic_words:
                    topic_labels[topic] = " ".join([word for word, _ in topic_words])
                else:
                    topic_labels[topic] = "No Topic"
            else:
                topic_labels[topic] = "No Topic"

        df["Bertopic"] = [topic_labels[topic] for topic in topics]
        status.update(label="BERTopic clustering complete!", state="complete")

    return df, topic_model

def create_2d_embeddings(embeddings, n_neighbors=15, min_dist=0.1, metric='cosine'):
    """Reduce embeddings to 2D using UMAP"""
    with st.status("Creating 2D visualization...", expanded=True) as status:
        st.write("Reducing dimensions with UMAP...")
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=2,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )
        embeddings_2d = umap_model.fit_transform(embeddings)
        status.update(label="2D visualization ready!", state="complete")
    return embeddings_2d

def plot_cluster_scatter(df, embeddings_2d, cluster_column, title):
    """Create interactive scatter plot for clusters"""
    plot_df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'cluster': df[cluster_column].astype(str),
        'keyword': df['keywords']
    })

    # Create color map
    unique_clusters = plot_df['cluster'].unique()
    n_clusters = len(unique_clusters)

    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color='cluster',
        hover_data=['keyword'],
        title=title,
        labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'cluster': 'Cluster'},
        color_discrete_sequence=px.colors.qualitative.Set3 if n_clusters <= 12 else None
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white')))
    fig.update_layout(
        height=600,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )

    return fig

def plot_cluster_distribution(df, cluster_column, title):
    """Create bar chart showing cluster distribution"""
    cluster_counts = df[cluster_column].value_counts().sort_values(ascending=True)

    fig = go.Figure(data=[
        go.Bar(
            y=[f"Cluster {c}" for c in cluster_counts.index],
            x=cluster_counts.values,
            orientation='h',
            marker=dict(
                color=cluster_counts.values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Keywords")
            ),
            text=cluster_counts.values,
            textposition='auto',
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Number of Keywords",
        yaxis_title="Cluster",
        height=max(400, len(cluster_counts) * 30),
        template='plotly_white',
        showlegend=False
    )

    return fig

def plot_top_keywords_per_cluster(df, cluster_column, top_n=5):
    """Show top keywords for each cluster"""
    cluster_data = []

    for cluster in df[cluster_column].unique():
        cluster_df = df[df[cluster_column] == cluster]
        top_keywords = cluster_df['keywords'].head(top_n).tolist()
        # Join keywords with comma separator
        keywords_str = ", ".join(top_keywords)
        cluster_data.append({
            "Cluster": f"Cluster {cluster}",
            "Sample Keywords": keywords_str
        })

    return pd.DataFrame(cluster_data)

def create_bertopic_visualizations(topic_model, df):
    """Create BERTopic-specific visualizations"""
    visualizations = {}

    try:
        # Topic bar chart
        with st.status("Generating BERTopic visualizations...", expanded=True) as status:
            st.write("Creating topic bar chart...")
            visualizations['barchart'] = topic_model.visualize_barchart(top_n_topics=10, n_words=8)

            st.write("Creating intertopic distance map...")
            visualizations['topics'] = topic_model.visualize_topics()

            st.write("Creating topic hierarchy...")
            visualizations['hierarchy'] = topic_model.visualize_hierarchy()

            status.update(label="BERTopic visualizations ready!", state="complete")
    except Exception as e:
        st.warning(f"Some BERTopic visualizations couldn't be generated: {str(e)}")

    return visualizations

# Main app
st.markdown("---")

# File upload
uploaded_file = st.file_uploader(
    "Upload Excel file with keywords",
    type=["xlsx", "xls"],
    help="Upload an Excel file containing your keywords"
)

if uploaded_file is not None:
    try:
        # Read Excel file
        excel_data = pd.ExcelFile(uploaded_file)

        # Sheet selection
        sheet_name = st.selectbox("Select sheet", excel_data.sheet_names)

        # Read selected sheet
        sheet_data = excel_data.parse(sheet_name)

        # Column selection
        column_name = st.selectbox("Select keyword column", sheet_data.columns)

        # Show preview
        st.subheader("Preview")
        st.dataframe(sheet_data[[column_name]].head(10), use_container_width=True)

        # Process button
        if st.button("🚀 Start Clustering", type="primary", use_container_width=True):
            try:
                # Create dataframe
                dataframe = sheet_data[[column_name]].copy()
                dataframe.columns = ["keywords"]
                dataframe = dataframe.dropna()

                st.info(f"Processing {len(dataframe)} keywords...")

                # Get stopwords
                stop_words = get_stopwords(language)

                # Clean keywords (minimal processing, preserves Vietnamese diacritics)
                dataframe = clean_keywords(dataframe, stop_words)

                # Apply Vietnamese word segmentation (premium feature)
                if language == "Vietnamese":
                    segmented_keywords = apply_vietnamese_word_segmentation(
                        dataframe["keywords"].tolist(),
                        language
                    )
                    dataframe["segmented"] = segmented_keywords
                else:
                    dataframe["segmented"] = dataframe["keywords"]

                # PREMIUM MODE: Use OpenAI embeddings
                if use_premium and openai_api_key:
                    st.info("🌟 Using Premium Mode with OpenAI embeddings")

                    # Get OpenAI embeddings
                    openai_embeddings = get_openai_embeddings(
                        dataframe["segmented"].tolist(),
                        openai_api_key
                    )

                    if openai_embeddings is not None:
                        # Store embeddings
                        dataframe["openai_emb"] = list(openai_embeddings)

                        # Cluster using OpenAI embeddings
                        dataframe = cluster_with_hdbscan(
                            dataframe,
                            "openai_emb",
                            "Premium Cluster",
                            clustering_blocks
                        )

                        # Generate labels with GPT-4o
                        gpt4_labels = generate_cluster_labels_with_gpt4(
                            dataframe,
                            "Premium Cluster",
                            openai_api_key,
                            language
                        )

                        if gpt4_labels:
                            dataframe["Cluster Label"] = dataframe["Premium Cluster"].map(gpt4_labels)
                        else:
                            dataframe["Cluster Label"] = "Label generation failed"

                        bertopic_model = None  # Skip BERTopic in premium mode
                    else:
                        st.warning("OpenAI embeddings failed, falling back to free mode...")
                        use_premium = False

                # FREE MODE: Use Word2Vec + BERTopic
                if not use_premium or not openai_api_key:
                    # Train Word2Vec directly on segmented keywords
                    dataframe_temp = dataframe.copy()
                    dataframe_temp["keywords"] = dataframe["segmented"]
                    dataframe_temp, w2v_model = train_word2vec(dataframe_temp)
                    dataframe["w2v"] = dataframe_temp["w2v"]

                    # Cluster Word2Vec
                    dataframe = cluster_with_hdbscan(dataframe, "w2v", "W2V Cluster", clustering_blocks)

                    # Load embedding model and calculate BERTopic
                    embedding_model = load_embedding_model(embedding_options[selected_embedding])
                    dataframe, bertopic_model = calculate_bertopic_clusters(dataframe, embedding_model, language)

                # Show results
                if use_premium and openai_api_key and "Premium Cluster" in dataframe.columns:
                    st.success("✅ Premium clustering complete! 🌟")
                else:
                    st.success("✅ Clustering complete!")

                # Display results
                st.subheader("Results Preview")
                if use_premium and openai_api_key and "Premium Cluster" in dataframe.columns:
                    display_cols = ["keywords", "segmented", "Premium Cluster", "Cluster Label"]
                else:
                    display_cols = ["keywords", "segmented", "W2V Cluster", "Bertopic"]
                st.dataframe(dataframe[display_cols].head(20), use_container_width=True)

                # Cluster statistics
                col1, col2, col3 = st.columns(3)
                if use_premium and openai_api_key and "Premium Cluster" in dataframe.columns:
                    with col1:
                        premium_clusters = len(dataframe["Premium Cluster"].unique())
                        st.metric("Premium Clusters (OpenAI)", premium_clusters)
                    with col2:
                        unique_labels = len(dataframe["Cluster Label"].unique())
                        st.metric("Unique Labels (GPT-4o)", unique_labels)
                    with col3:
                        total_keywords = len(dataframe)
                        st.metric("Total Keywords", total_keywords)
                else:
                    with col1:
                        w2v_clusters = len(dataframe["W2V Cluster"].unique())
                        st.metric("Word2Vec Clusters", w2v_clusters)
                    with col2:
                        bertopic_clusters = len(dataframe["Bertopic"].unique())
                        st.metric("BERTopic Clusters", bertopic_clusters)
                    with col3:
                        total_keywords = len(dataframe)
                        st.metric("Total Keywords", total_keywords)

                # Visualizations
                st.markdown("---")
                st.subheader("📊 Cluster Visualizations")

                # Determine which embedding to visualize
                if use_premium and openai_api_key and "openai_emb" in dataframe.columns:
                    # Premium mode visualizations
                    embeddings = np.array(dataframe["openai_emb"].tolist())
                    embeddings_2d = create_2d_embeddings(embeddings)
                    cluster_column = "Premium Cluster"
                    title_prefix = "Premium (OpenAI)"

                    viz_tabs = st.tabs([
                        "🗺️ Premium Scatter",
                        "📊 Cluster Distribution",
                        "📝 Top Keywords & Labels"
                    ])
                else:
                    # Free mode visualizations
                    embeddings = np.array(dataframe["w2v"].tolist())
                    embeddings_2d = create_2d_embeddings(embeddings)
                    cluster_column = "W2V Cluster"
                    title_prefix = "Word2Vec"

                    viz_tabs = st.tabs([
                        "🗺️ Word2Vec Scatter",
                        "📊 Distribution",
                        "📝 Top Keywords",
                        "🎯 BERTopic Visuals"
                    ])

                with viz_tabs[0]:
                    st.markdown(f"### {title_prefix} Cluster Scatter Plot")
                    st.markdown("Interactive 2D visualization using UMAP dimensionality reduction")
                    scatter = plot_cluster_scatter(
                        dataframe,
                        embeddings_2d,
                        cluster_column,
                        f"{title_prefix} Clusters (UMAP 2D Projection)"
                    )
                    st.plotly_chart(scatter, use_container_width=True)

                with viz_tabs[1]:
                    st.markdown("### Cluster Size Distribution")
                    st.markdown("Number of keywords in each cluster")

                    if use_premium and openai_api_key and "Premium Cluster" in dataframe.columns:
                        # Premium mode: Show cluster distribution with GPT-4o labels
                        dist = plot_cluster_distribution(
                            dataframe,
                            "Premium Cluster",
                            "Premium Cluster Distribution (OpenAI)"
                        )
                        st.plotly_chart(dist, use_container_width=True)
                    else:
                        # Free mode: Show both W2V and BERTopic
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("#### Word2Vec Clusters")
                            w2v_dist = plot_cluster_distribution(
                                dataframe,
                                "W2V Cluster",
                                "Word2Vec Cluster Distribution"
                            )
                            st.plotly_chart(w2v_dist, use_container_width=True)

                        with col_b:
                            st.markdown("#### BERTopic Clusters")
                            bertopic_counts = dataframe["Bertopic"].value_counts().sort_values(ascending=True)
                            fig_bertopic = go.Figure(data=[
                                go.Bar(
                                    y=[label[:50] + "..." if len(label) > 50 else label for label in bertopic_counts.index],
                                    x=bertopic_counts.values,
                                    orientation='h',
                                    marker=dict(
                                        color=bertopic_counts.values,
                                        colorscale='Plasma',
                                        showscale=True,
                                        colorbar=dict(title="Keywords")
                                    ),
                                    text=bertopic_counts.values,
                                    textposition='auto',
                                )
                            ])
                            fig_bertopic.update_layout(
                                title="BERTopic Cluster Distribution",
                                xaxis_title="Number of Keywords",
                                yaxis_title="Topic",
                                height=max(400, len(bertopic_counts) * 30),
                                template='plotly_white',
                                showlegend=False
                            )
                            st.plotly_chart(fig_bertopic, use_container_width=True)

                with viz_tabs[2]:
                    if use_premium and openai_api_key and "Premium Cluster" in dataframe.columns:
                        st.markdown("### Premium Clusters with GPT-4o Labels")
                        st.markdown("AI-generated cluster labels and sample keywords")

                        # Show clusters with their GPT-4o labels
                        premium_clusters = []
                        for cluster_id in sorted(dataframe["Premium Cluster"].unique()):
                            cluster_df = dataframe[dataframe["Premium Cluster"] == cluster_id]
                            label = cluster_df["Cluster Label"].iloc[0]
                            keywords = cluster_df["keywords"].head(10).tolist()
                            premium_clusters.append({
                                "Cluster": cluster_id,
                                "GPT-4o Label": label,
                                "Count": len(cluster_df),
                                "Sample Keywords": ", ".join(keywords)
                            })

                        premium_df = pd.DataFrame(premium_clusters)
                        st.dataframe(premium_df, use_container_width=True, hide_index=True)

                    else:
                        st.markdown("### Top Keywords per Cluster")
                        st.markdown("Sample keywords from each cluster")

                        col_x, col_y = st.columns(2)

                        with col_x:
                            st.markdown("#### Word2Vec Clusters")
                            w2v_top = plot_top_keywords_per_cluster(dataframe, "W2V Cluster", top_n=10)
                            st.dataframe(w2v_top, use_container_width=True, hide_index=True)

                        with col_y:
                            st.markdown("#### BERTopic Clusters")
                            bertopic_examples = []
                            for topic in dataframe["Bertopic"].unique()[:20]:  # Limit to 20 topics
                                topic_df = dataframe[dataframe["Bertopic"] == topic]
                                examples = ", ".join(topic_df["keywords"].head(5).tolist())
                                bertopic_examples.append({
                                    "Topic": topic[:60] + "..." if len(topic) > 60 else topic,
                                    "Sample Keywords": examples[:100] + "..." if len(examples) > 100 else examples
                                })
                            bertopic_df = pd.DataFrame(bertopic_examples)
                            st.dataframe(bertopic_df, use_container_width=True, hide_index=True)

                # Only show BERTopic advanced visualizations in free mode
                if not (use_premium and openai_api_key and "Premium Cluster" in dataframe.columns):
                    with viz_tabs[3]:
                        st.markdown("### BERTopic Advanced Visualizations")
                        st.markdown("Semantic topic analysis and relationships")

                        bertopic_viz = create_bertopic_visualizations(bertopic_model, dataframe)

                        if 'barchart' in bertopic_viz:
                            st.markdown("#### Top Words per Topic")
                            st.plotly_chart(bertopic_viz['barchart'], use_container_width=True)

                        if 'topics' in bertopic_viz:
                            st.markdown("#### Intertopic Distance Map")
                            st.markdown("Shows how topics relate to each other in semantic space")
                            st.plotly_chart(bertopic_viz['topics'], use_container_width=True)

                        if 'hierarchy' in bertopic_viz:
                            st.markdown("#### Topic Hierarchy")
                            st.markdown("#### Hierarchical clustering of topics")
                            st.plotly_chart(bertopic_viz['hierarchy'], use_container_width=True)

                st.markdown("---")

                # Download button
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    dataframe.to_excel(writer, index=False, sheet_name='Clusters')
                output.seek(0)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                st.download_button(
                    label="📥 Download Results",
                    data=output,
                    file_name=f"cluster_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.exception(e)

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Upload an Excel file to get started")

    # Instructions
    with st.expander("📖 How to use"):
        st.markdown("""
        1. **Upload** an Excel file containing your keywords
        2. **Select** the sheet and column with your keywords
        3. **Choose** language and model settings in the sidebar
        4. **Click** "Start Clustering" to begin processing
        5. **Download** the results with all clustering data

        ### Output Columns
        - **keywords**: Original keywords (Vietnamese diacritics preserved!)
        - **w2v**: Word2Vec embedding vectors (300 dimensions)
        - **W2V Cluster**: Word2Vec cluster ID
        - **Bertopic**: Semantic topic description
        """)

    with st.expander("⚙️ Technical Details"):
        st.markdown("""
        ### Clustering Methods
        - **Word2Vec**: Custom-trained embeddings on your dataset
        - **BERTopic**: Transformer-based semantic clustering using HDBSCAN

        ### Language Support
        - **Vietnamese**: Fully supported! Diacritics are preserved throughout processing
        - **English**: Fully supported via multilingual models

        ### Processing Pipeline
        1. Minimal preprocessing (preserves original text)
        2. Word2Vec training on your keywords
        3. HDBSCAN clustering (automatic cluster detection)
        4. BERTopic semantic topic modeling

        ### Models Used
        - Sentence Transformers for embeddings
        - HDBSCAN for clustering
        - Snowball Stemmer for text normalization
        """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Powered by BERTopic and SentenceTransformers")
