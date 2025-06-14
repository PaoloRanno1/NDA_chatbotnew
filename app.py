import streamlit as st
import os
import tempfile
from typing import Dict, Any, List
import json
from datetime import datetime

# Import your NDA analyzer class (assuming it's in the same directory or installed as a package)
from NDA_chatbot import EnhancedNDAAnalyzer

# Page configuration
st.set_page_config(
    page_title="Strada NDA Analyzer Chatbot",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }

    .chat-message {
        display: flex;
        margin-bottom: 1.5rem;
        width: 100%;
    }

    .message-content {
        max-width: 75%;
        padding: 1rem 1.25rem;
        border-radius: 20px;
        color: #262730;
        font-weight: 400;
        line-height: 1.5;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        word-wrap: break-word;
    }

    .user-message {
        justify-content: flex-start;
    }

    .user-message .message-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-left-radius: 8px;
        margin-right: auto;
    }

    .assistant-message {
        justify-content: flex-end;
    }

    .assistant-message .message-content {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-bottom-right-radius: 8px;
        margin-left: auto;
        color: #2d3748;
    }

    .message-label {
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .user-message .message-label {
        color: rgba(255,255,255,0.9);
    }

    .assistant-message .message-label {
        color: #64748b;
    }

    /* Markdown content styling within messages */
    .assistant-message .message-content h1,
    .assistant-message .message-content h2,
    .assistant-message .message-content h3 {
        color: #1a202c;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .assistant-message .message-content p {
        margin-bottom: 0.75rem;
    }

    .assistant-message .message-content ul,
    .assistant-message .message-content ol {
        margin-left: 1rem;
        margin-bottom: 0.75rem;
    }

    .assistant-message .message-content strong {
        color: #2d3748;
        font-weight: 600;
    }

    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .status-success {
        color: #4CAF50;
        font-weight: bold;
    }

    .status-error {
        color: #f44336;
        font-weight: bold;
    }

    .status-info {
        color: #2196F3;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    if 'document_name' not in st.session_state:
        st.session_state.document_name = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None


def display_chat_history():
    """Display the chat history with modern chat layout"""
    if st.session_state.chat_history:
        st.subheader("💬 Conversation History")

        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                # User message - left aligned
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-content">
                        <div class="message-label">You</div>
                        {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                # Assistant message - right aligned with markdown support
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-content">
                        <div class="message-label">AI Assistant</div>
                """, unsafe_allow_html=True)

                # Create a custom container for markdown content
                col1, col2 = st.columns([1, 3])  # This creates space on the left
                with col2:
                    # Display markdown content in a styled container
                    st.markdown(f"""
                    <div style="
                        background-color: #ffffff;
                        border: 1px solid #e2e8f0;
                        border-radius: 15px;
                        padding: 1rem;
                        margin-bottom: 1rem;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                    ">
                    """, unsafe_allow_html=True)

                    # Render the markdown content
                    st.markdown(message['content'])

                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("</div></div>", unsafe_allow_html=True)

                # Show sources if available
                if 'sources' in message and message['sources']:
                    col1, col2 = st.columns([1, 3])
                    with col2:
                        with st.expander(f"📚 Sources ({len(message['sources'])} documents)", expanded=False):
                            for j, source in enumerate(message['sources']):
                                st.text(f"Source {j + 1}: {source.page_content[:200]}...")


def main():
    # Initialize session state
    initialize_session_state()

    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>📋 NDA Analyzer Chatbot</h1>
        <p>Intelligent Analysis of Non-Disclosure Agreements</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for configuration and controls
    with st.sidebar:
        st.header("🔧 Configuration")

        # OpenAI API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use the analyzer"
        )

        # Model selection
        model_choice = st.selectbox(
            "Select Model",
            ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            index=0,
            help="Choose the OpenAI model for analysis"
        )

        st.divider()

        # Document upload section
        st.header("📁 Document Management")

        uploaded_file = st.file_uploader(
            "Upload NDA Document",
            type=['pdf'],
            help="Upload a PDF file of the NDA to analyze"
        )

        # Initialize analyzer and load document
        if uploaded_file is not None and api_key:
            if st.button("🚀 Initialize Analyzer", type="primary"):
                with st.spinner("Initializing analyzer..."):
                    try:
                        # Initialize analyzer
                        st.session_state.analyzer = EnhancedNDAAnalyzer(
                            openai_api_key=api_key,
                            model_name=model_choice
                        )

                        # Save and load document
                        temp_path = save_uploaded_file(uploaded_file)
                        if temp_path:
                            success = st.session_state.analyzer.load_nda_document(temp_path)
                            if success:
                                st.session_state.document_loaded = True
                                st.session_state.document_name = uploaded_file.name
                                st.success("✅ Analyzer initialized and document loaded!")
                                # Clean up temp file
                                os.unlink(temp_path)
                            else:
                                st.error("❌ Failed to load document")
                        else:
                            st.error("❌ Failed to save uploaded file")
                    except Exception as e:
                        st.error(f"❌ Error initializing analyzer: {str(e)}")

        # Document status
        if st.session_state.document_loaded:
            st.markdown(f"""
            <div class="sidebar-info">
                <p class="status-success">✅ Document Loaded</p>
                <p><strong>File:</strong> {st.session_state.document_name}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sidebar-info">
                <p class="status-error">❌ No Document Loaded</p>
                <p>Please upload a PDF and initialize the analyzer</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Quick actions
        st.header("⚡ Quick Actions")

        if st.session_state.document_loaded and st.session_state.analyzer:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📄 Summary", use_container_width=True):
                    st.session_state.quick_action = "summary"

            with col2:
                if st.button("⚖️ Legal Analysis", use_container_width=True):
                    st.session_state.quick_action = "legal_analysis"

            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                if st.session_state.analyzer:
                    st.session_state.analyzer.clear_memory()
                st.rerun()

        st.divider()

        # Conversation summary
        if st.session_state.analyzer and st.session_state.chat_history:
            st.header("📊 Chat Statistics")
            st.metric("Messages", len(st.session_state.chat_history))

            # Intent distribution
            intents = [msg.get('intent', 'Unknown') for msg in st.session_state.chat_history if
                       msg['role'] == 'assistant']
            if intents:
                from collections import Counter
                intent_counts = Counter(intents)
                st.write("**Intent Distribution:**")
                for intent, count in intent_counts.items():
                    st.write(f"• {intent}: {count}")

    # Main content area
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to get started.")
        st.info("""
        **How to get started:**
        1. Enter your OpenAI API key in the sidebar
        2. Upload a PDF document of the NDA
        3. Click 'Initialize Analyzer'
        4. Start chatting with the analyzer!
        """)
        return

    if not st.session_state.document_loaded:
        st.info("📁 Please upload an NDA document in the sidebar to begin analysis.")
        return

    # Handle quick actions
    if hasattr(st.session_state, 'quick_action'):
        if st.session_state.quick_action == "summary":
            with st.spinner("Generating document summary..."):
                result = st.session_state.analyzer.chat("Please provide a summary of this NDA document")
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': 'Please provide a summary of this NDA document',
                    'timestamp': datetime.now()
                })
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['response'],
                    'intent': result['intent'],
                    'sources': result.get('sources', []),
                    'timestamp': datetime.now()
                })
        elif st.session_state.quick_action == "legal_analysis":
            with st.spinner("Performing legal analysis..."):
                result = st.session_state.analyzer.chat(
                    "Please perform a detailed legal compliance analysis of this NDA")
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': 'Please perform a detailed legal compliance analysis of this NDA',
                    'timestamp': datetime.now()
                })
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['response'],
                    'intent': result['intent'],
                    'sources': result.get('sources', []),
                    'timestamp': datetime.now()
                })

        # Clear the quick action
        delattr(st.session_state, 'quick_action')
        st.rerun()

    # Chat interface
    st.header("💬 Chat with NDA Analyzer")

    # Display chat history
    display_chat_history()

    # Chat input (must be outside any container)
    user_input = st.chat_input("Ask me anything about the NDA document...")

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })

        # Get response from analyzer
        with st.spinner("Analyzing your question..."):
            try:
                result = st.session_state.analyzer.chat(user_input)

                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['response'],
                    'intent': result['intent'],
                    'sources': result.get('sources', []),
                    'timestamp': datetime.now()
                })

            except Exception as e:
                st.error(f"Error processing your request: {str(e)}")
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': f"I encountered an error: {str(e)}",
                    'intent': 'ERROR',
                    'sources': [],
                    'timestamp': datetime.now()
                })

        # Rerun to display new messages
        st.rerun()

    # Example questions
    if not st.session_state.chat_history:
        st.subheader("💡 Example Questions")

        example_questions = [
            "What are the main parties involved in this NDA?",
            "What are the confidentiality obligations?",
            "How long does this agreement last?",
            "Are there any restrictions on the use of confidential information?",
            "What happens if the agreement is breached?",
            "Does this NDA comply with our firm's requirements?",
            "Are there any concerning clauses I should be aware of?"
        ]

        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question, key=f"example_{i}"):
                    # Simulate clicking the question
                    st.session_state.example_question = question
                    st.rerun()

    # Handle example question clicks
    if hasattr(st.session_state, 'example_question'):
        user_input = st.session_state.example_question

        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })

        # Get response from analyzer
        with st.spinner("Analyzing your question..."):
            try:
                result = st.session_state.analyzer.chat(user_input)

                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['response'],
                    'intent': result['intent'],
                    'sources': result.get('sources', []),
                    'timestamp': datetime.now()
                })

            except Exception as e:
                st.error(f"Error processing your request: {str(e)}")

        # Clear the example question
        delattr(st.session_state, 'example_question')
        st.rerun()


if __name__ == "__main__":
    main()
