import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
import time
from datetime import datetime
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fitz  # PyMuPDF for better PDF extraction
import io

# Page configuration
st.set_page_config(
    page_title="Buddy Note",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .stats-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .local-badge {
        background-color: #28a745;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-left: 10px;
    }
    

</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'document_text' not in st.session_state:
    st.session_state.document_text = ""
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None


# Improved PDF text extraction
def extract_text_from_pdf(file):
    """Extract text from PDF using multiple methods for reliability"""
    text = ""

    try:
        # Method 1: Try PyMuPDF (fitz) first - most reliable
        try:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            file.seek(0)  # Reset file pointer
            if text.strip():
                return text
        except:
            pass

        # Method 2: Try PyPDF2 as fallback
        try:
            file.seek(0)
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            file.seek(0)
            if text.strip():
                return text
        except:
            pass

        # Method 3: If both fail, return empty text
        return text

    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""


# Local text processing functions
def simple_text_search(query, text_chunks, top_k=3):
    """Local text search using TF-IDF similarity"""
    if not text_chunks:
        return []

    # Combine query with chunks for TF-IDF
    all_texts = [query] + text_chunks

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Calculate similarity between query and chunks
        query_vector = tfidf_matrix[0]
        chunk_vectors = tfidf_matrix[1:]

        similarities = cosine_similarity(query_vector, chunk_vectors).flatten()

        # Get top k most similar chunks
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = [(text_chunks[i], similarities[i]) for i in top_indices if similarities[i] > 0]

        return results
    except:
        # Fallback to simple keyword matching
        results = []
        for i, chunk in enumerate(text_chunks):
            if query.lower() in chunk.lower():
                results.append((chunk, 0.5))  # Default similarity score
        return results[:top_k]


def generate_local_response(query, relevant_chunks):
    """Generate response using local logic without API calls"""
    if not relevant_chunks:
        return "I couldn't find relevant information in the document for your question. Try rephrasing or ask about something else."

    # Create a context-based response
    context = "\n\n".join([f"📄 Passage {i + 1} (Relevance: {score:.2f}):\n{chunk}"
                           for i, (chunk, score) in enumerate(relevant_chunks)])

    response = f"""**🔍 I found relevant information in your document:**

{context}

**💡 Summary:**
Based on the document content, here's what I found related to your question. The most relevant passages are shown above with their similarity scores.

**📊 Found {len(relevant_chunks)} relevant passages**"""

    return response


def extract_key_phrases(text, num_phrases=5):
    """Extract key phrases using TF-IDF"""
    if not text.strip():
        return ["document", "content", "information", "text", "data"]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=100)
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()

        # Get top phrases
        top_indices = scores.argsort()[-num_phrases:][::-1]
        return [feature_names[i] for i in top_indices]
    except:
        return ["document", "content", "information", "text", "data"]


# Header
st.markdown('<h1 class="main-header">📚 Buddy Note</h1>',
            unsafe_allow_html=True)
st.markdown("### 🚀 Your AI-powered document assistant - No APIs, No Internet required!")

# Sidebar
with st.sidebar:
    st.title("⚙️ Control Panel")

    # File upload with enhanced UI
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    file = st.file_uploader("📁 Upload Your PDF Document", type="pdf", help="Upload your notes or textbook PDF")
    st.markdown('</div>', unsafe_allow_html=True)

    if file is not None:
        st.session_state.uploaded_file = file

    # Document statistics
    if st.session_state.document_processed:
        st.markdown("### 📊 Document Stats")
        text_length = len(st.session_state.document_text)
        word_count = len(st.session_state.document_text.split())
        chunk_count = len(st.session_state.text_chunks)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Words", f"{word_count:,}")
            st.metric("Chunks", chunk_count)
        with col2:
            st.metric("Characters", f"{text_length:,}")
            if st.session_state.uploaded_file:
                try:
                    pdf_reader = PdfReader(st.session_state.uploaded_file)
                    st.metric("Pages", len(pdf_reader.pages))
                except:
                    st.metric("Pages", "N/A")

    # Settings
    st.markdown("### ⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 100, 500, 250, help="Size of text chunks for processing")
    similarity_results = st.slider("Similarity Results", 1, 10, 3, help="Number of relevant passages to show")

    # Processing options
    st.markdown("### 🔧 Processing Options")
    use_advanced_search = st.checkbox("Use advanced search", value=True, help="Uses TF-IDF for better results")

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Export chat
    if st.session_state.chat_history:
        if st.button("💾 Export Chat"):
            chat_json = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                label="Download Chat History",
                data=chat_json,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Chat interface
    st.markdown("### 💬 Chat with Your Document")

    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message['content']}
                <br><small>{message['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>NoteBote:</strong> {message['content']}
                <br><small>{message['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)

    # Chat input
    user_query = st.chat_input("Ask something about your document...")

    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_query,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })

        # Process query
        if st.session_state.document_processed and st.session_state.text_chunks:
            with st.spinner("🔍 Searching document locally..."):
                try:
                    # Use local search instead of vector store
                    if use_advanced_search:
                        relevant_chunks = simple_text_search(user_query, st.session_state.text_chunks,
                                                             top_k=similarity_results)
                    else:
                        # Simple keyword search fallback
                        relevant_chunks = []
                        for chunk in st.session_state.text_chunks:
                            if user_query.lower() in chunk.lower():
                                relevant_chunks.append((chunk, 1.0))
                        relevant_chunks = relevant_chunks[:similarity_results]

                    # Generate response using local logic
                    response = generate_local_response(user_query, relevant_chunks)

                    # Add bot response to chat history
                    st.session_state.chat_history.append({
                        'role': 'bot',
                        'content': response,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })

                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing your query: {str(e)}")
        else:
            st.warning("Please upload and process a document first.")

with col2:
    # Document information panel
    st.markdown("### 📋 Document Info")

    if st.session_state.uploaded_file is not None:
        file = st.session_state.uploaded_file

        if not st.session_state.document_processed:
            with st.spinner("📄 Processing document locally..."):
                try:
                    # Extract text from PDF using improved method
                    text = extract_text_from_pdf(file)

                    if not text.strip():
                        st.error("❌ Could not extract text from the PDF. The file might be scanned or protected.")
                        st.info("💡 Try with a different PDF file that contains selectable text.")
                    else:
                        st.session_state.document_text = text

                        # Break into chunks
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=50
                        )
                        chunks = text_splitter.split_text(text)
                        st.session_state.text_chunks = chunks

                        # Try to create embeddings (optional)
                        try:
                            embeddings = HuggingFaceEmbeddings(
                                model_name="all-MiniLM-L6-v2",
                                model_kwargs={'device': 'cpu'},
                                encode_kwargs={'normalize_embeddings': False}
                            )
                            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                        except Exception as e:
                            st.session_state.vector_store = None
                            st.info("ℹ️ Using fast local search instead of embeddings")

                        st.session_state.document_processed = True

                        st.success("✅ Document processed successfully!")
                        st.balloons()

                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
                    st.info("💡 Make sure you're uploading a valid PDF file.")

        if st.session_state.document_processed:
            # Document preview
            with st.expander("👁️ Document Preview", expanded=False):
                if st.session_state.document_text:
                    preview_text = st.session_state.document_text[:800] + "..." if len(
                        st.session_state.document_text) > 800 else st.session_state.document_text
                    st.text_area("Extracted Text", preview_text, height=150, label_visibility="collapsed")
                else:
                    st.warning("No text extracted from document")

            # Key phrases
            with st.expander("🔑 Key Phrases", expanded=False):
                key_phrases = extract_key_phrases(st.session_state.document_text)
                st.write("Top phrases in your document:")
                for phrase in key_phrases:
                    st.write(f"• {phrase}")

            # Quick actions
            st.markdown("### ⚡ Quick Actions")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📖 Summarize"):
                    with st.spinner("Generating local summary..."):
                        # Simple local summary
                        words = st.session_state.document_text.split()
                        summary = f"**Document Overview:**\n\n"
                        summary += f"• Contains {len(words):,} words across {len(st.session_state.text_chunks)} sections\n"
                        summary += f"• Main topics: {', '.join(extract_key_phrases(st.session_state.document_text, 3))}\n"
                        summary += f"• First few lines: {st.session_state.document_text[:200]}..."
                        st.info(summary)

            with col2:
                if st.button("❓ Suggest Questions"):
                    suggested_questions = [
                        "What are the main topics covered?",
                        "Can you show me relevant sections about key concepts?",
                        "What are the important points in this document?",
                        "What conclusions or summaries are presented?"
                    ]
                    st.write("💡 Try asking:")
                    for q in suggested_questions:
                        st.write(f"- {q}")




    else:
        st.info("👆 Upload a PDF document to get started!")
        st.markdown("""
        <div class="success-box">
        <strong>🎯 100% Local Features:</strong>
        <ul>
            <li>✅ No API keys required</li>
            <li>✅ No internet connection needed</li>
            <li>✅ Complete privacy</li>
            <li>✅ Fast local processing</li>
            <li>✅ Document Insights</li>
            <li>✅ Privacy allocated</li>
            
            
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>AI generated, reference only </p>
</div>
""", unsafe_allow_html=True)
