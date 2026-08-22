%%writefile app.py

from tenacity import retry, wait_random_exponential, stop_after_attempt
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests
from bs4 import BeautifulSoup
import re
from sentence_transformers import SentenceTransformer
import faiss
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import google.generativeai as genai

# --- NLTK Data Downloads ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')

# --- Constants ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 100 # Overlap to maintain context between chunks

# --- Model Architectures ---
def create_stage1_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """Defines the architecture for the Stage 1 binary classification model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_stage2_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes_val=None):
    """Defines the architecture for the Stage 2 multi-class classification model."""
    if num_classes_val is None:
        raise ValueError("num_classes_val must be provided.")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(num_classes_val, activation='softmax')
    ])
    return model

# --- RAG Helper Function: Text Chunking ---
def chunk_text(text, chunk_size, chunk_overlap):
    """Splits a given text into overlapping chunks for RAG processing."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
        if start >= len(text):
            break
    return chunks

# --- Gemini API Setup for Summarization ---
# This part assumes the API key is set as an environment variable or a Streamlit secret
try:
    gemini_api_key = os.getenv('GOOGLE_API_KEY')
    if not gemini_api_key:
        st.error("GOOGLE_API_KEY not found. Please set it as an environment variable or Streamlit secret.")
    else:
        genai.configure(api_key=gemini_api_key)
        # Initialize Gemini model once using st.cache_resource
        @st.cache_resource
        def load_gemini_model():
            return genai.GenerativeModel('gemini-flash-latest')
        gemini_model = load_gemini_model()

except Exception as e:
    st.error(f"Error configuring Gemini API for summarization: {e}")
    gemini_model = None # Set to None if API fails to configure

@retry(wait=wait_random_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def generate_summary_with_gemini(text_chunk):
    """Generates a brief summary of a text chunk using the Gemini API with retry logic."""
    if gemini_model is None:
        return "[LLM unavailable for summarization]"
    try:
        prompt = f"Summarize the following text briefly for a user interested in recycling or composting advice. Focus on key actions or information: {text_chunk}"
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.warning(f"Error calling Gemini API for summarization: {e}")
        return "[Error generating summary]"

# --- Cached Function to Load All Models and RAG Data ---
@st.cache_resource
def load_models_and_rag_data():
    """Loads and caches all ML models and RAG data for the Streamlit app."""
    # Instantiate Stage 1 Model and Load Weights
    stage1_model_loaded = create_stage1_model()
    stage1_model_loaded.load_weights('best_stage1_model.weights.h5')

    # Instantiate Stage 2 Model and Load Weights
    num_classes_garbage = 10 # Hardcoded based on notebook's output
    stage2_model_loaded = create_stage2_model(num_classes_val=num_classes_garbage)
    stage2_model_loaded.load_weights('best_stage2_model.weights.h5')

    # Load Sentence Transformer model
    embedding_model_loaded = SentenceTransformer('all-MiniLM-L6-v2')

    # --- RAG Data Processing ---
    web_urls = [
        {'name': 'EPA Composting', 'url': 'https://www.epa.gov/recycle/composting-home'},
        {'name': 'EPA Recycling', 'url': 'https://www.epa.gov/recycle/recycling-basics-and-benefits'}
    ]

    web_text_data_cleaned = {}
    for item in web_urls:
        doc_name = item['name']
        try:
            response = requests.get(item['url'])
            response.raise_for_status() # Raise an exception for HTTP errors
            html_content = response.text

            soup = BeautifulSoup(html_content, 'html.parser')
            for script_or_style in soup(['script', 'style']):
                script_or_style.extract()
            text = soup.get_text()
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                web_text_data_cleaned[doc_name] = text
        except requests.exceptions.RequestException as e:
            st.warning(f"Could not download or parse content for {doc_name}: {e}")
        except Exception as e:
            st.warning(f"Error processing web content for {doc_name}: {e}")

    all_chunks = []
    chunk_metadata = [] # (doc_name, chunk_index)
    for doc_name, text_content in web_text_data_cleaned.items():
        doc_chunks = chunk_text(text_content, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(doc_chunks)
        chunk_metadata.extend([(doc_name, i) for i in range(len(doc_chunks))])

    chunk_embeddings = np.array([])
    faiss_index = None

    if all_chunks:
        chunk_embeddings = embedding_model_loaded.encode(all_chunks, show_progress_bar=False)
        embedding_dimension = chunk_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        faiss_index.add(np.array(chunk_embeddings))

    # Define class labels for the models
    stage1_class_labels = ['biodegradable', 'non-biodegradable']
    stage2_class_labels = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']

    return stage1_model_loaded, stage2_model_loaded, embedding_model_loaded, faiss_index, all_chunks, chunk_metadata, stage1_class_labels, stage2_class_labels

# Load all necessary components when the app starts
stage1_model, stage2_model, embedding_model, faiss_index, all_chunks, chunk_metadata, stage1_class_labels, stage2_class_labels = load_models_and_rag_data()

# --- RAG Helper Function: Retrieval ---
def retrieve_info_from_rag(query, top_k=3):
    """Retrieves relevant text chunks from the FAISS index based on a query."""
    if faiss_index is None:
        return []
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)

    retrieved_info = []
    for i, idx in enumerate(indices[0]):
        if idx < len(all_chunks): # Safety check for valid index
            retrieved_info.append({
                'text': all_chunks[idx],
                'source_doc': chunk_metadata[idx][0],
                'distance': distances[0][i]
            })
    return retrieved_info

# --- Combined Classification and RAG Function ---
def classify_and_get_rag_info_streamlit(pil_image):
    """Classifies an image and retrieves RAG information with summaries for Streamlit display."""
    # Preprocess the uploaded image
    img = pil_image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Rescale to [0, 1]

    # Stage 1 Classification (Biodegradable vs. Non-Biodegradable)
    stage1_pred_raw = stage1_model.predict(img_array, verbose=0)
    biodegradable_score = stage1_pred_raw[0][0]

    classification_result = {}
    rag_query = ""

    if biodegradable_score < 0.5: # Biodegradable
        main_category = stage1_class_labels[0]
        main_category_confidence = 1 - biodegradable_score
        rag_query = "how to compost food scraps or biodegradable waste"
    else: # Non-Biodegradable
        main_category = stage1_class_labels[1]
        main_category_confidence = biodegradable_score

        # Stage 2 Classification (Specific Garbage Type) for non-biodegradable
        stage2_pred = stage2_model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(stage2_pred[0])
        specific_type = stage2_class_labels[predicted_class_index]
        specific_type_confidence = stage2_pred[0][predicted_class_index]
        rag_query = f"how to recycle {specific_type} waste"

        classification_result['specific_type'] = specific_type
        classification_result['specific_type_confidence'] = specific_type_confidence

    classification_result['main_category'] = main_category
    classification_result['main_category_confidence'] = main_category_confidence

    # Retrieve RAG information
    retrieved_rag_info = retrieve_info_from_rag(rag_query, top_k=2)

    # Generate summaries for retrieved info
    summarized_rag_info = []
    for item in retrieved_rag_info:
        summary = generate_summary_with_gemini(item['text'])
        summarized_rag_info.append({
            'summary': summary,
            'source_doc': item['source_doc']
        })

    classification_result['rag_info'] = summarized_rag_info
    classification_result['rag_query'] = rag_query

    return classification_result

# --- Streamlit Application UI ---
st.set_page_config(page_title="Garbage Classification & Recycling Advisor", layout="centered")

st.title("🗑️ Garbage Classification & Recycling Advisor")
st.markdown("Upload an image of a waste item. I'll classify it and provide relevant recycling or composting advice from EPA sources.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button("Classify & Get Advice"):
        with st.spinner("Classifying and fetching and summarizing advice..."):
            results = classify_and_get_rag_info_streamlit(image_display)

            st.subheader("Classification Results:")
            if results['main_category'] == 'biodegradable':
                st.success(f"This item is likely **Biodegradable** (Confidence: {results['main_category_confidence']:.2f}).")
                st.markdown("--- # Composting Advice")
                st.subheader(f"Composting Advice for Biodegradable Waste:")
            else:
                st.info(f"This item is likely **Non-Biodegradable** (Confidence: {results['main_category_confidence']:.2f}).")
                if results.get('specific_type'): # Check if specific type was classified
                    st.write(f"Specifically, it appears to be **{results['specific_type'].replace('_', ' ').title()}** (Confidence: {results['specific_type_confidence']:.2f}).")
                st.markdown("--- # Recycling Advice")
                st.subheader(f"Recycling Advice for {results['main_category'].capitalize()} Waste:")

            # Display RAG information (now with summaries)
            if results['rag_info']:
                st.markdown(f"**Relevant information related to '{results['rag_query']}' from EPA sources:**")
                for i, res in enumerate(results['rag_info']):
                    st.markdown(f"- **Source:** {res['source_doc']}")
                    st.markdown(f"  **Summary:** {res['summary']}")
            else:
                st.warning("No specific relevant information found from EPA sources for this query.")

            st.markdown("--- # Your Feedback")

    st.subheader("Help Us Improve!")
    st.markdown("Your feedback helps us make this tool better. Please let us know what you think.")

    # Feedback inputs
    classification_feedback = st.text_area(
        "**Classification Accuracy:** Was the item classified correctly? If not, what was it?",
        key="clf_feedback"
    )
    rag_feedback = st.text_area(
        "**Advice Usefulness:** Was the recycling/composting advice helpful and relevant to your needs?",
        key="rag_feedback"
    )

    if st.button("Submit Feedback"):
        if classification_feedback or rag_feedback:
            st.success("Thank you for your valuable feedback! We appreciate your input.")
        else:
            st.warning("Please provide some feedback in at least one of the boxes before submitting.")
