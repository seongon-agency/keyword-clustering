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

def reassign_outliers_to_nearest_cluster(embeddings, clusters):
    """Reassign outlier points to their nearest cluster"""
    from sklearn.metrics.pairwise import cosine_similarity

    outlier_indices = np.where(clusters == -1)[0]
    non_outlier_indices = np.where(clusters != -1)[0]

    if len(outlier_indices) == 0 or len(non_outlier_indices) == 0:
        return clusters  # No outliers or no clusters to assign to

    # Calculate cosine similarity between outliers and non-outliers
    outlier_embeddings = embeddings[outlier_indices]
    non_outlier_embeddings = embeddings[non_outlier_indices]

    similarities = cosine_similarity(outlier_embeddings, non_outlier_embeddings)

    # For each outlier, find the nearest non-outlier and assign its cluster
    new_clusters = clusters.copy()
    for i, outlier_idx in enumerate(outlier_indices):
        nearest_non_outlier_idx = non_outlier_indices[np.argmax(similarities[i])]
        new_clusters[outlier_idx] = clusters[nearest_non_outlier_idx]

    return new_clusters

def cluster_with_hdbscan(df, embedding_column, output_column, n_blocks):
    """Cluster embeddings using HDBSCAN in blocks with outlier reassignment"""
    with st.status(f"Clustering {output_column}...", expanded=True) as status:
        embeddings = np.array(df[embedding_column].tolist())

        # Relaxed HDBSCAN parameters to reduce outliers
        hdbscan_params = {
            'min_cluster_size': 2,
            'min_samples': 1,
            'cluster_selection_epsilon': 0.1,  # Merge clusters within this distance
            'cluster_selection_method': 'leaf',  # More lenient than 'eom'
            'core_dist_n_jobs': -1
        }

        # For small datasets, don't use blocking
        min_block_size = 10  # HDBSCAN needs at least this many samples
        if len(embeddings) < min_block_size * 2:
            # Just cluster everything at once
            st.write(f"Clustering {len(embeddings)} keywords...")
            clusterer = hdbscan.HDBSCAN(**hdbscan_params)
            all_clusters = clusterer.fit_predict(embeddings)

            # Count initial outliers
            initial_outliers = np.sum(all_clusters == -1)
            st.write(f"Initial outliers: {initial_outliers}")

            # Reassign outliers to nearest cluster
            if initial_outliers > 0:
                st.write("Reassigning outliers to nearest clusters...")
                all_clusters = reassign_outliers_to_nearest_cluster(embeddings, all_clusters)
                final_outliers = np.sum(all_clusters == -1)
                st.write(f"Final outliers: {final_outliers}")

            df[output_column] = all_clusters
            status.update(label=f"{output_column} clustering complete!", state="complete")
            return df

        # Ensure n_blocks doesn't create blocks smaller than min_block_size
        max_blocks = len(embeddings) // min_block_size
        n_blocks = min(n_blocks, max_blocks)

        block_size = len(embeddings) // n_blocks
        all_clusters = np.full(len(embeddings), -1, dtype=int)

        st.write(f"Processing {n_blocks} blocks...")
        progress_bar = st.progress(0)
        for i in range(n_blocks):
            start = i * block_size
            end = (i + 1) * block_size if i < n_blocks - 1 else len(embeddings)
            block = embeddings[start:end]

            clusterer = hdbscan.HDBSCAN(**hdbscan_params)
            block_clusters = clusterer.fit_predict(block)
            all_clusters[start:end] = block_clusters

            progress_bar.progress((i + 1) / n_blocks)

        # Count initial outliers
        initial_outliers = np.sum(all_clusters == -1)
        st.write(f"Initial outliers: {initial_outliers}")

        # Reassign outliers to nearest cluster
        if initial_outliers > 0:
            st.write("Reassigning outliers to nearest clusters...")
            all_clusters = reassign_outliers_to_nearest_cluster(embeddings, all_clusters)
            final_outliers = np.sum(all_clusters == -1)
            st.write(f"Final outliers: {final_outliers}")

        df[output_column] = all_clusters
        status.update(label=f"{output_column} clustering complete!", state="complete")

    return df

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
            # Require API key
            if not openai_api_key:
                st.error("⚠️ OpenAI API key is required! Please add it to .env file or enter it in the sidebar.")
                st.stop()

            try:
                # Create dataframe
                dataframe = sheet_data[[column_name]].copy()
                dataframe.columns = ["keywords"]
                dataframe = dataframe.dropna()

                st.info(f"🌟 Processing {len(dataframe)} keywords with Premium AI...")

                # Basic deduplication (preserve Vietnamese diacritics)
                with st.status("Preprocessing keywords...", expanded=True) as status:
                    original_count = len(dataframe)
                    dataframe["keywords"] = dataframe["keywords"].str.strip()
                    dataframe["keywords"] = dataframe["keywords"].str.replace(r"\s+", " ", regex=True)
                    dataframe = dataframe.drop_duplicates(subset=["keywords"], keep="first").reset_index(drop=True)
                    dataframe = dataframe[dataframe["keywords"].str.strip() != ""]
                    st.write(f"Removed {original_count - len(dataframe)} duplicates")
                    status.update(label="Preprocessing complete!", state="complete")

                # Apply Vietnamese word segmentation
                if language == "Vietnamese":
                    segmented_keywords = apply_vietnamese_word_segmentation(
                        dataframe["keywords"].tolist(),
                        language
                    )
                    dataframe["segmented"] = segmented_keywords
                else:
                    dataframe["segmented"] = dataframe["keywords"]

                # Get OpenAI embeddings
                openai_embeddings = get_openai_embeddings(
                    dataframe["segmented"].tolist(),
                    openai_api_key
                )

                if openai_embeddings is None:
                    st.error("Failed to get OpenAI embeddings. Please check your API key and try again.")
                    st.stop()

                # Store embeddings
                dataframe["openai_emb"] = list(openai_embeddings)

                # Cluster using HDBSCAN
                dataframe = cluster_with_hdbscan(
                    dataframe,
                    "openai_emb",
                    "Cluster",
                    clustering_blocks
                )

                # Generate cluster labels with GPT-4o
                gpt4_labels = generate_cluster_labels_with_gpt4(
                    dataframe,
                    "Cluster",
                    openai_api_key,
                    language
                )

                if gpt4_labels:
                    dataframe["Cluster Label"] = dataframe["Cluster"].map(gpt4_labels)
                else:
                    dataframe["Cluster Label"] = "Unlabeled"

                # Show results
                st.success("✅ Premium AI clustering complete! 🌟")

                # Display results
                st.subheader("Results Preview")
                display_cols = ["keywords", "segmented", "Cluster", "Cluster Label"]
                st.dataframe(dataframe[display_cols].head(20), use_container_width=True)

                # Cluster statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    num_clusters = len(dataframe["Cluster"].unique())
                    st.metric("AI Clusters (OpenAI)", num_clusters)
                with col2:
                    unique_labels = len(dataframe["Cluster Label"].unique())
                    st.metric("Unique Labels (GPT-4o)", unique_labels)
                with col3:
                    total_keywords = len(dataframe)
                    st.metric("Total Keywords", total_keywords)

                # Visualizations
                st.markdown("---")
                st.subheader("📊 AI Cluster Visualizations")

                # Create visualizations
                embeddings = np.array(dataframe["openai_emb"].tolist())
                embeddings_2d = create_2d_embeddings(embeddings)

                viz_tabs = st.tabs([
                    "🗺️ Cluster Map",
                    "📊 Distribution",
                    "📝 Labels & Keywords"
                ])

                with viz_tabs[0]:
                    st.markdown("### AI Cluster Scatter Plot (OpenAI Embeddings)")
                    st.markdown("Interactive 2D visualization using UMAP dimensionality reduction")
                    scatter = plot_cluster_scatter(
                        dataframe,
                        embeddings_2d,
                        "Cluster",
                        "AI Clusters (UMAP 2D Projection)"
                    )
                    st.plotly_chart(scatter, use_container_width=True)

                with viz_tabs[1]:
                    st.markdown("### Cluster Size Distribution")
                    st.markdown("Number of keywords in each AI-generated cluster")
                    dist = plot_cluster_distribution(
                        dataframe,
                        "Cluster",
                        "AI Cluster Distribution (OpenAI + GPT-4o)"
                    )
                    st.plotly_chart(dist, use_container_width=True)

                with viz_tabs[2]:
                    st.markdown("### AI Clusters with GPT-4o Labels")
                    st.markdown("GPT-4o generated cluster labels and sample keywords")

                    # Show clusters with their GPT-4o labels
                    cluster_data = []
                    for cluster_id in sorted(dataframe["Cluster"].unique()):
                        cluster_df = dataframe[dataframe["Cluster"] == cluster_id]
                        label = cluster_df["Cluster Label"].iloc[0]
                        keywords = cluster_df["keywords"].head(10).tolist()
                        cluster_data.append({
                            "Cluster": cluster_id,
                            "GPT-4o Label": label,
                            "Count": len(cluster_df),
                            "Sample Keywords": ", ".join(keywords)
                        })

                    cluster_summary = pd.DataFrame(cluster_data)
                    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

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
        1. **Add API Key**: Place your OpenAI API key in a `.env` file or enter it in the sidebar
        2. **Upload** an Excel file containing your keywords
        3. **Select** the sheet and column with your keywords
        4. **Choose** language (Vietnamese or English)
        5. **Click** "Start Clustering" to begin AI-powered processing
        6. **Download** the results with AI-generated clusters and labels

        ### Output Columns
        - **keywords**: Original keywords (Vietnamese diacritics preserved!)
        - **segmented**: Word-segmented version (Vietnamese only)
        - **openai_emb**: OpenAI text-embedding-3-large vectors (3072 dimensions)
        - **Cluster**: AI-generated cluster ID (HDBSCAN)
        - **Cluster Label**: GPT-4o generated descriptive label
        """)

    with st.expander("⚙️ Technical Details"):
        st.markdown("""
        ### Premium AI Pipeline
        - **OpenAI text-embedding-3-large**: State-of-the-art 3072-dimensional embeddings
        - **HDBSCAN**: Density-based clustering for automatic cluster detection
        - **GPT-4o**: Intelligent cluster label generation

        ### Language Support
        - **Vietnamese**: Full support with word segmentation (underthesea) and diacritics preservation
        - **English**: Full support via OpenAI multilingual embeddings

        ### Processing Pipeline
        1. Vietnamese word segmentation (if selected)
        2. OpenAI embeddings generation (batch processing)
        3. HDBSCAN clustering in configurable blocks
        4. GPT-4o cluster label generation

        ### Cost Estimate
        - Approximately $0.50-$2.00 per 1000 keywords
        - Embeddings: ~$0.13 per 1M tokens
        - GPT-4o labels: ~$0.05 per cluster
        """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Powered by OpenAI & GPT-4o")
