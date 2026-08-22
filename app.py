import streamlit as st
import numpy as np # Keep numpy for array operations on image
from PIL import Image
import os
import requests
from bs4 import BeautifulSoup
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import google.generativeai as genai
from tenacity import retry, wait_random_exponential, stop_after_attempt
import io # Needed for BytesIO for image processing

# --- NLTK Data Downloads ---
# NLTK downloads should ideally happen outside the main app run flow if possible,
# or be handled gracefully. On Streamlit Community Cloud, these will download once
# into the app environment.
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("test") # 'punkt' is needed for word_tokenize
except LookupError:
    nltk.download('punkt')

# --- Constants ---
# IMG_HEIGHT and IMG_WIDTH still useful for resizing images before sending to Gemini Vision
IMG_HEIGHT = 128
IMG_WIDTH = 128

# --- Gemini API Setup ---
gemini_text_model = None
gemini_vision_model = None

@st.cache_resource
def load_gemini_api_config(api_key):
    genai.configure(api_key=api_key)
    # The user specified gemini-2.5-flash for vision.
    # gemini-flash-latest is good for general text.
    text_model = genai.GenerativeModel('gemini-flash-latest')
    vision_model = genai.GenerativeModel('gemini-2.5-flash')
    return text_model, vision_model

try:
    gemini_api_key = os.getenv('GOOGLE_API_KEY')
    if not gemini_api_key:
        st.error("GOOGLE_API_KEY not found. Please set it as an environment variable or Streamlit secret.")
    else:
        gemini_text_model, gemini_vision_model = load_gemini_api_config(gemini_api_key)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. Image classification and advice generation will be disabled.")


@retry(wait=wait_random_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def generate_summary_with_gemini(text_chunk):
    """Generates a brief summary of a text chunk using the Gemini API text model with retry logic."""
    if gemini_text_model is None:
        return "[LLM unavailable for summarization]"
    try:
        # Adjusted prompt for a more direct summary, as it's not for RAG retrieval anymore
        prompt = f"Summarize the following text briefly, focusing on key actions or information for recycling or composting: {text_chunk}"
        response = gemini_text_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.warning(f"Error calling Gemini API for summarization: {e}")
        return "[Error generating summary]"


# --- RAG Helper Function: Advice Generation with Gemini Text Model ---
@retry(wait=wait_random_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def generate_rag_advice_with_gemini(query, web_data_cleaned):
    """Generates recycling/composting advice using Gemini's text capabilities based on cleaned web data."""
    if gemini_text_model is None:
        return {"summary": "[LLM unavailable for advice generation]", "source_doc": "N/A"}

    # Combine all web text into a single context for Gemini.
    # For very large datasets, this might exceed context window.
    # A more advanced RAG would select specific relevant documents.
    full_context = "\n\n".join([f"--- Document: {name} ---\n{text}" for name, text in web_data_cleaned.items()])

    # Formulate a prompt to guide Gemini in providing relevant advice
    prompt = f"""Based on the following context documents and the query '{query}', provide concise and actionable recycling or composting advice.
    For each piece of advice, clearly indicate which document it came from (e.g., 'From EPA Composting: ...').

    Context Documents:
    {full_context}

    Query: {query}

    Advice:
    """
    try:
        response = gemini_text_model.generate_content(prompt)
        # Gemini's response will be the advice, sources are expected to be embedded by Gemini itself
        return {"summary": response.text, "source_doc": "Multiple EPA Sources"} # Gemini should cite in its response
    except Exception as e:
        st.warning(f"Error generating RAG advice with Gemini: {e}")
        return {"summary": "[Error generating advice]", "source_doc": "N/A"}


# --- Cached Function to Load RAG Data ---
@st.cache_resource
def load_rag_data():
    """Downloads and processes web content for RAG, caching the result."""
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

    # Define class labels directly, as models are no longer loaded locally
    stage1_class_labels = ['biodegradable', 'non-biodegradable']
    stage2_class_labels = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']

    return web_text_data_cleaned, stage1_class_labels, stage2_class_labels

# Load RAG data and class labels when the app starts
web_text_data_cleaned, stage1_class_labels, stage2_class_labels = load_rag_data()


# --- Combined Classification and RAG Function (Streamlit) ---
def classify_and_get_rag_info_streamlit(pil_image):
    """Classifies an image using Gemini Vision and generates RAG advice using Gemini Text."""
    if gemini_vision_model is None or gemini_text_model is None:
        return {
            'main_category': "Error",
            'specific_type': "Error",
            'rag_info': [{'summary': "[Gemini API not configured]", 'source_doc': "N/A"}],
            'rag_query': ""
        }

    # Prepare image for Gemini Vision API
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format='JPEG') # Save PIL Image to bytes
    image_parts = [{"mime_type": "image/jpeg", "data": img_bytes.getvalue()}]

    # --- Image Classification using Gemini Vision Model ---
    classification_prompt = [
        "Analyze the image and classify the waste. First, state if it is 'biodegradable' or 'non-biodegradable'. "
        "Then, if non-biodegradable, identify the specific type from these categories: 'battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash'. "
        "Provide the output as two separate classifications on new lines, exactly in the format: 'Overall category: [category]' and 'Specific type: [type]' (if applicable, otherwise omit 'Specific type')."
        "Example 1: 'Overall category: biodegradable'"
        "Example 2: 'Overall category: non-biodegradable\nSpecific type: plastic'",
        image_parts[0]
    ]

    try:
        vision_response = gemini_vision_model.generate_content(classification_prompt)
        classification_text = vision_response.text.strip()
    except Exception as e:
        st.error(f"Error classifying image with Gemini Vision API: {e}")
        classification_text = "Overall category: unknown" # Fallback

    main_category = "unknown"
    specific_type = "unknown"
    rag_query = ""

    # Parse classification text
    lines = classification_text.split('\n')
    for line in lines:
        if line.startswith("Overall category:"):
            main_category = line.split(":", 1)[1].strip().lower()
        elif line.startswith("Specific type:"):
            specific_type = line.split(":", 1)[1].strip().lower()

    if main_category == 'biodegradable':
        rag_query = "how to compost food scraps or biodegradable waste"
    elif main_category == 'non-biodegradable' and specific_type != 'unknown':
        rag_query = f"how to recycle {specific_type} waste"
    else:
        rag_query = "general recycling and composting advice"

    # --- Generate RAG Advice using Gemini Text Model ---
    rag_advice = generate_rag_advice_with_gemini(rag_query, web_text_data_cleaned)

    classification_result = {
        'main_category': main_category,
        # Confidence is not directly available from Gemini's generative response
        'main_category_confidence': "N/A",
        'specific_type': specific_type if specific_type != 'unknown' else None,
        'specific_type_confidence': "N/A", # Not directly available
        'rag_info': [rag_advice], # Now rag_advice is a dictionary
        'rag_query': rag_query
    }

    return classification_result

# --- Streamlit Application UI ---
st.set_page_config(page_title="Garbage Classification & Recycling Advisor", layout="centered")

st.title("🗑️ Garbage Classification & Recycling Advisor")
st.markdown("Upload an image of a waste item. I'll classify it and provide relevant recycling or composting advice from EPA sources using advanced AI.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button("Classify & Get Advice"):
        with st.spinner("Classifying image and fetching advice..."):
            results = classify_and_get_rag_info_streamlit(image_display)

            st.subheader("Classification Results:")
            if results['main_category'] == 'biodegradable':
                st.success(f"This item is likely **Biodegradable**.")
                st.markdown("---")
                st.subheader(f"Composting Advice:")
            elif results['main_category'] == 'non-biodegradable':
                st.info(f"This item is likely **Non-Biodegradable**.")
                if results.get('specific_type') and results['specific_type'] != 'unknown':
                    st.write(f"Specifically, it appears to be **{results['specific_type'].replace('_', ' ').title()}**.")
                st.markdown("---")
                st.subheader(f"Recycling Advice:")
            else:
                 st.warning(f"Could not confidently classify the item.")
                 st.markdown("---")
                 st.subheader(f"General Advice:")


            # Display RAG information (now with summaries)
            if results['rag_info']:
                st.markdown(f"**Relevant information related to '{results['rag_query']}' from EPA sources:**")
                for i, res in enumerate(results['rag_info']): # This loop will only run once now
                    st.markdown(f"- **Source:** {res['source_doc']}")
                    st.markdown(f"  **Advice:** {res['summary']}")
            else:
                st.warning("No specific relevant information found from EPA sources for this query.")

            st.markdown("---")
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
