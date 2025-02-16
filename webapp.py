import gdown
import os
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from langdetect import detect
from googletrans import Translator

# Download necessary NLTK resources
nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)

# Filepath for GloVe embeddings
GLOVE_FILE = "glove.6B.100d.txt"

# Google Drive file ID for GloVe (replace with your actual file ID)
GLOVE_FILE_ID = "192I71mfi3SwHiTf7HP40_aUbFHe-trMX"

# Function to download GloVe embeddings
def download_glove():
    if not os.path.exists(GLOVE_FILE):
        st.info("🔄 Downloading GloVe embeddings... This may take a few minutes.")
        url = f"https://drive.google.com/uc?id={GLOVE_FILE_ID}"
        gdown.download(url, GLOVE_FILE, quiet=False)
        st.success("✅ Download complete!")

# Download if file is missing
download_glove()

# Load GloVe word embeddings
word_embeddings = {}
try:
    with open(GLOVE_FILE, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            word_embeddings[word] = coefs
    st.success("✅ GloVe embeddings loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Error: '{GLOVE_FILE}' not found! Download failed or check the file path.")

# Function to remove stopwords
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join([w for w in word_tokens if w.lower() not in stop_words])
    return filtered_text

# Function to remove punctuations
def remove_punctuations(text):
    translator = str.maketrans('', '', punctuation)
    text_without_punctuations = text.translate(translator)
    return text_without_punctuations

# Function to convert text to lowercase
def convert_lower(text):
    return text.lower()

# Function to generate summary
def generate_summary(text):
    if not text.strip():
        return []

    lower_text = convert_lower(text)
    new_text = remove_stop_words(lower_text)
    sentences = sent_tokenize(text)
    cleaned_sentences = sent_tokenize(remove_punctuations(new_text))

    sentence_vectors = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        if words:
            sentence_vector = np.mean([word_embeddings.get(word, np.zeros((100,))) for word in words], axis=0)
        else:
            sentence_vector = np.zeros((100,))
        sentence_vectors.append(sentence_vector)

    # Create similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, -1), sentence_vectors[j].reshape(1, -1))[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    return [s[1] for s in ranked_sentences[:5]]

# Streamlit UI
def main():
    st.title("📝 Text Summarization & Translation App")
    
    text = st.text_area("Enter your text here:")

    lang_options = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'hi': 'Hindi',
        'ru': 'Russian', 'zh-cn': 'Chinese (Simplified)', 'ja': 'Japanese', 'ar': 'Arabic'
    }

    trans_lang = st.selectbox("Select language for translation:", list(lang_options.values()))

    if st.button("Summarize"):
        if not text.strip():
            st.warning("⚠️ Please enter some text to summarize.")
            return

        # Generate summary
        summarized_sentences = generate_summary(text)

        if not summarized_sentences:
            st.warning("❌ No summary generated. Try providing more meaningful text.")
            return

        # Display original & summarized text
        st.subheader("📌 Original Text:")
        st.write(text)

        st.subheader("🔹 Summarized Text:")
        for sentence in summarized_sentences:
            st.write(f"- {sentence}")

        # Detect language
        try:
            detected_lang = detect(text)
            st.write(f"🌍 Detected Language: {detected_lang}")
        except Exception as e:
            st.warning(f"Language detection failed: {e}")

        # Translation
        lang_code = [k for k, v in lang_options.items() if v == trans_lang][0]
        translator = Translator()
        translated_sentences = []

        for sentence in summarized_sentences:
            try:
                translated_sentence = translator.translate(sentence, dest=lang_code).text
                translated_sentences.append(translated_sentence)
            except Exception as e:
                st.error(f"Translation failed for '{sentence}' with error: {e}")

        # Display Translated Summary
        st.subheader(f"🌎 Translated Summary ({trans_lang}):")
        for translated_sentence in translated_sentences:
            st.write(f"- {translated_sentence}")

if __name__ == "__main__":
    main()
