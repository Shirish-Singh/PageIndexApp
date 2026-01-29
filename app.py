"""
PageIndex Document QA - Streamlit Web Application

A web application for document Q&A using PageIndex's tree-structured indexing
and LLM-powered reasoning.

Usage:
    streamlit run app.py
"""

import streamlit as st
import tempfile
import os
import json
import sys
from io import BytesIO
from pathlib import Path

# Check for required dependencies
def check_dependencies():
    """Check if all required dependencies are installed."""
    missing = []
    try:
        import PyPDF2
    except ImportError:
        missing.append("PyPDF2")
    try:
        import pymupdf
    except ImportError:
        missing.append("pymupdf")
    try:
        import tiktoken
    except ImportError:
        missing.append("tiktoken")
    try:
        import openai
    except ImportError:
        missing.append("openai")
    try:
        import groq
    except ImportError:
        missing.append("groq")

    if missing:
        st.error(f"""
        **Missing Dependencies:** {', '.join(missing)}

        Please install them by running:
        ```bash
        pip install {' '.join(missing)}
        ```

        Or install all requirements:
        ```bash
        pip install -r requirements.txt
        ```

        **If using PyCharm:** Make sure the interpreter is set to the project's `.venv`:
        - Go to Settings → Project → Python Interpreter
        - Select: `.venv/bin/python`
        """)
        st.stop()

check_dependencies()

# Add PageIndex to path
sys.path.insert(0, str(Path(__file__).parent / "PageIndex"))

# Import and configure LLM provider BEFORE importing PageIndex
from llm_provider import (
    set_provider,
    get_provider,
    get_available_models,
    chat_completion,
    patch_pageindex_for_provider,
    patch_tiktoken,
    get_pageindex_model,
    LLMConfig,
    PROVIDERS
)

# Apply patches early
patch_tiktoken()


def init_session_state():
    """Initialize session state variables."""
    if "index_data" not in st.session_state:
        st.session_state.index_data = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    if "provider_initialized" not in st.session_state:
        st.session_state.provider_initialized = False
    # Store provider config in session state for persistence
    if "current_provider" not in st.session_state:
        st.session_state.current_provider = "groq"
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "current_api_key" not in st.session_state:
        st.session_state.current_api_key = None


def test_api_connection(provider_name: str, model: str, api_key: str = None) -> tuple:
    """Test if the API connection works with the given config."""
    try:
        # Temporarily configure the provider
        kwargs = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        set_provider(provider_name, **kwargs)
        patch_pageindex_for_provider()

        # Try a simple API call
        response = chat_completion(
            "Say 'OK' and nothing else.",
            temperature=0,
            max_tokens=10
        )
        if response and len(response) > 0:
            return True, "Connection successful!"
        return False, "Empty response from API"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return False, "Invalid API key"
        elif "429" in error_msg or "rate" in error_msg.lower():
            return True, "API key valid (rate limited, but working)"
        else:
            return False, f"Connection error: {error_msg[:100]}"


def configure_provider(provider_name: str, model: str, api_key: str = None, test_connection: bool = True):
    """Configure the LLM provider."""
    try:
        kwargs = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key

        # Test connection first if requested
        if test_connection:
            success, message = test_api_connection(provider_name, model, api_key)
            if not success:
                st.error(f"API test failed: {message}")
                return False
            st.success(message)

        # Apply configuration
        set_provider(provider_name, **kwargs)
        patch_pageindex_for_provider()

        # Store in session state
        st.session_state.provider_initialized = True
        st.session_state.current_provider = provider_name
        st.session_state.current_model = model
        if api_key:
            st.session_state.current_api_key = api_key

        return True
    except Exception as e:
        st.error(f"Failed to configure provider: {e}")
        return False


def build_index(uploaded_file) -> dict:
    """Build PageIndex tree structure from uploaded file."""
    from pageindex import page_index

    # Get original filename
    original_filename = uploaded_file.name

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # Get the appropriate model for the current provider
        model = get_pageindex_model()
        st.info(f"Using model: {model}")

        with st.spinner("Building document index... This may take a few minutes."):
            index_data = page_index(
                tmp_path,
                model=model,
                if_add_node_summary="yes",
                if_add_node_text="yes",
                if_add_node_id="yes"
            )

        # Fix the document name to use original filename instead of temp path
        if isinstance(index_data, dict):
            index_data["doc_name"] = original_filename
        elif isinstance(index_data, list):
            # If it returns a list, wrap it in a dict with proper doc_name
            index_data = {
                "doc_name": original_filename,
                "structure": index_data
            }

        return index_data
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def format_tree_structure(structure: list, indent: int = 0) -> str:
    """Format tree structure for display."""
    lines = []
    for node in structure:
        prefix = "  " * indent + ("|- " if indent > 0 else "")
        title = node.get("title", "Untitled")
        pages = f"[p.{node.get('start_index', '?')}-{node.get('end_index', '?')}]"
        lines.append(f"{prefix}{title} {pages}")

        if "nodes" in node and node["nodes"]:
            lines.append(format_tree_structure(node["nodes"], indent + 1))

    return "\n".join(lines)


def get_relevant_sections(structure: list, query: str, max_sections: int = 5) -> list:
    """
    Use LLM to identify relevant sections for the query.
    Returns list of relevant node summaries and text.
    """
    # Flatten structure to get all nodes
    def flatten_nodes(nodes, parent_path=""):
        result = []
        for node in nodes:
            path = f"{parent_path}/{node.get('title', '')}" if parent_path else node.get('title', '')
            result.append({
                "node_id": node.get("node_id"),
                "title": node.get("title"),
                "path": path,
                "summary": node.get("summary", ""),
                "text": node.get("text", ""),
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index")
            })
            if "nodes" in node and node["nodes"]:
                result.extend(flatten_nodes(node["nodes"], path))
        return result

    all_nodes = flatten_nodes(structure)

    # Create a compact representation for LLM to reason about
    nodes_summary = []
    for node in all_nodes:
        nodes_summary.append({
            "id": node["node_id"],
            "title": node["title"],
            "summary": node["summary"][:200] if node["summary"] else ""
        })

    # Ask LLM to identify relevant sections
    prompt = f"""You are analyzing a document to find relevant sections for a query.

Document Sections:
{json.dumps(nodes_summary, indent=2)}

Query: {query}

Identify the {max_sections} most relevant section IDs that would help answer this query.
Return ONLY a JSON array of section IDs, e.g., ["0001", "0003", "0005"]
"""

    try:
        response = chat_completion(prompt, temperature=0.1)
        # Parse response to get section IDs
        import re
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            relevant_ids = json.loads(match.group())
            relevant_nodes = [n for n in all_nodes if n["node_id"] in relevant_ids]
            return relevant_nodes
    except Exception as e:
        st.warning(f"Error identifying sections: {e}")

    # Fallback: return first few nodes
    return all_nodes[:max_sections]


def answer_query(query: str, index_data: dict, chat_history: list) -> str:
    """Generate answer using tree-structured reasoning."""
    structure = index_data.get("structure", [])
    doc_name = index_data.get("doc_name", "Document")

    # Get relevant sections
    relevant_sections = get_relevant_sections(structure, query)

    # Build context from relevant sections
    context_parts = []
    for section in relevant_sections:
        section_text = section.get("text", "")
        if section_text:
            # Clean up the text - remove null bytes and control characters
            clean_text = ''.join(c for c in section_text if c.isprintable() or c in '\n\t')
            context_parts.append(f"""
--- SECTION: {section['title']} (Pages {section['start_index']}-{section['end_index']}) ---
{clean_text[:4000]}
--- END SECTION ---
""")

    if not context_parts:
        return "I don't have enough context from the document to answer this question. The document may not have been indexed properly."

    context = "\n".join(context_parts)

    # Build conversation history
    history_text = ""
    if chat_history:
        recent_history = chat_history[-6:]  # Last 3 exchanges
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    system_prompt = f"""You are a document assistant. Answer questions ONLY using the document content provided below.

DOCUMENT NAME: {doc_name}

DOCUMENT CONTENT:
{context}

STRICT RULES:
1. ONLY use information from the DOCUMENT CONTENT above
2. Do NOT make up or invent any information
3. Do NOT create fake sections, page numbers, or details
4. If something is not in the document, say "This is not mentioned in the document"
5. Keep your answer concise and factual
6. Quote or paraphrase from the actual document text

{f"CONVERSATION HISTORY:{chr(10)}{history_text}" if history_text else ""}
"""

    # Generate response
    response = chat_completion(
        query,
        system_prompt=system_prompt,
        temperature=0.1,  # Lower temperature for more factual responses
        max_tokens=1024
    )

    return response


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <span style="font-size: 1.8rem;">⚙️</span>
                <h2 style="margin: 0.5rem 0 0 0; font-size: 1.3rem; font-weight: 600;">Configuration</h2>
            </div>
        """, unsafe_allow_html=True)

        # Get current/stored provider
        provider_options = list(PROVIDERS.keys())
        current_provider_idx = 0
        if st.session_state.current_provider in provider_options:
            current_provider_idx = provider_options.index(st.session_state.current_provider)

        # Provider selection
        provider = st.selectbox(
            "LLM Provider",
            options=provider_options,
            index=current_provider_idx,
            help="Select the LLM provider to use"
        )

        # Model selection - use stored model if same provider
        available_models = get_available_models(provider)
        current_model_idx = 0
        if st.session_state.current_model and st.session_state.current_provider == provider:
            if st.session_state.current_model in available_models:
                current_model_idx = available_models.index(st.session_state.current_model)

        model = st.selectbox(
            "Model",
            options=available_models,
            index=current_model_idx,
            help="Select the model to use"
        )

        # API Key input
        api_key_env = f"{provider.upper()}_API_KEY"
        existing_key = os.getenv(api_key_env, "")
        # Check if we have a stored key for this provider
        stored_key = st.session_state.current_api_key if st.session_state.current_provider == provider else None

        if stored_key:
            api_key_display = "****" + stored_key[-4:] + " (session)"
        elif existing_key:
            api_key_display = "****" + existing_key[-4:] + " (env)"
        else:
            api_key_display = ""

        api_key = st.text_input(
            f"API Key ({api_key_env})",
            value="",
            type="password",
            placeholder=api_key_display if api_key_display else "Enter API key",
            help=f"Enter your {provider.upper()} API key (leave empty to use existing)"
        )

        # Show current status
        if st.session_state.provider_initialized:
            st.caption(f"✓ Active: {st.session_state.current_provider} / {st.session_state.current_model or 'default'}")

        if st.button("Apply Configuration", type="primary"):
            # Use new key if provided, otherwise use stored key or env key
            key_to_use = api_key if api_key else (stored_key if stored_key else None)
            with st.spinner("Testing API connection..."):
                if configure_provider(provider, model, key_to_use, test_connection=True):
                    st.rerun()  # Refresh to show new status

        st.divider()

        # Document info
        if st.session_state.index_data:
            st.subheader("Document Info")
            doc_name = st.session_state.index_data.get("doc_name", "Unknown")
            st.write(f"**Name:** {doc_name}")

            structure = st.session_state.index_data.get("structure", [])
            st.write(f"**Sections:** {len(structure)}")

            if st.checkbox("Show Structure"):
                st.text(format_tree_structure(structure))

            # Download index button
            st.subheader("Export Index")
            index_json = json.dumps(st.session_state.index_data, indent=2, ensure_ascii=False)
            # Create a safe filename
            safe_name = doc_name.replace(".pdf", "").replace(" ", "_")[:50]
            st.download_button(
                label="Download Index (JSON)",
                data=index_json,
                file_name=f"{safe_name}_index.json",
                mime="application/json",
                help="Download the document index to use later without re-processing"
            )

        st.divider()

        # Clear chat
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        if st.button("Clear Document Index"):
            st.session_state.index_data = None
            st.session_state.document_name = None
            st.session_state.chat_history = []
            st.rerun()

        # About section
        st.divider()
        with st.expander("ℹ️ About"):
            st.markdown("""
            **PageIndex Document QA** is a free, open-source tool for intelligent document understanding.

            **How it works:**
            - 🌳 Builds a tree-structured index from your PDF
            - 🤖 Uses LLM to navigate and answer questions
            - ⚡ No vector database or embeddings needed

            **Privacy:**
            - 📁 Documents are processed locally
            - 🔒 Only relevant text chunks sent to LLM API
            - 🏠 Use Ollama for fully offline/private processing

            **Links:**
            - [GitHub Repository](https://github.com/Shirish-Singh/PageIndexApp)
            - [PageIndex Library](https://github.com/VectifyAI/PageIndex)

            ---
            *Made with ❤️ · Free to use*
            """)


def load_index_from_json(json_file) -> dict:
    """Load a previously saved index from JSON file."""
    try:
        content = json_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        index_data = json.loads(content)

        # Validate the index structure
        if not isinstance(index_data, dict):
            raise ValueError("Invalid index format: expected a JSON object")
        if "structure" not in index_data:
            raise ValueError("Invalid index format: missing 'structure' field")
        if "doc_name" not in index_data:
            raise ValueError("Invalid index format: missing 'doc_name' field")

        return index_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


def render_main_content():
    """Render the main content area."""
    # Hero Header
    st.markdown("""
        <div class="hero-header">
            <h1>📚 PageIndex Document QA</h1>
            <p>Upload PDFs and ask questions using AI-powered tree-structured indexing</p>
            <span class="hero-badge">Open Source</span>
        </div>
    """, unsafe_allow_html=True)

    # Feature highlights - collapsible
    with st.expander("✨ Why PageIndex? Click to learn more", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🌳</div>
                <div class="feature-title">Tree-Structured Indexing</div>
                <div class="feature-desc">
                    Unlike traditional RAG that chunks documents blindly, PageIndex builds a
                    hierarchical tree that preserves document structure - chapters, sections,
                    and their relationships.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🚀</div>
                <div class="feature-title">No Vector Database</div>
                <div class="feature-desc">
                    No embeddings, no vector stores, no complex setup. PageIndex uses the LLM
                    itself to navigate the document tree and find relevant sections intelligently.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🔒</div>
                <div class="feature-title">Privacy-Conscious</div>
                <div class="feature-desc">
                    Your document stays local. Only small text chunks are sent to the LLM
                    when answering questions - never the entire document.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("""
        **How it works:** When you ask a question, PageIndex navigates the document tree to find relevant sections,
        then sends only those specific text chunks to the LLM. This is more efficient and preserves context better
        than sending random chunks from vector similarity search.
        """)

    # Create tabs for different upload options
    tab_pdf, tab_json = st.tabs(["📄 Upload PDF", "📁 Load Saved Index"])

    with tab_pdf:
        # File uploader for PDF
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=["pdf"],
            help="Upload a PDF file to build a searchable index",
            key="pdf_uploader"
        )

        # Process uploaded PDF file
        if uploaded_file is not None:
            file_name = uploaded_file.name

            # Check if it's a new file
            if st.session_state.document_name != file_name:
                if not st.session_state.provider_initialized:
                    # Initialize with default provider
                    try:
                        from llm_provider import get_provider
                        get_provider()
                        patch_pageindex_for_provider()
                        st.session_state.provider_initialized = True
                    except Exception as e:
                        st.error(f"Please configure an LLM provider in the sidebar first. Error: {e}")
                        return

                st.info(f"Processing: {file_name}")

                try:
                    index_data = build_index(uploaded_file)
                    st.session_state.index_data = index_data
                    st.session_state.document_name = file_name
                    st.session_state.chat_history = []
                    st.success("Document indexed successfully!")
                    st.rerun()  # Rerun to show download button
                except Exception as e:
                    st.error(f"Error building index: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return

        # Show index info and download button when index exists
        if st.session_state.index_data and st.session_state.document_name and not st.session_state.document_name.startswith("json:"):
            doc_name = st.session_state.index_data.get("doc_name", "Document")
            num_sections = len(st.session_state.index_data.get("structure", []))

            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"**Indexed:** {doc_name} ({num_sections} sections)")
            with col2:
                index_json = json.dumps(st.session_state.index_data, indent=2, ensure_ascii=False)
                safe_name = doc_name.replace(".pdf", "").replace(" ", "_")[:50]
                st.download_button(
                    label="Download Index",
                    data=index_json,
                    file_name=f"{safe_name}_index.json",
                    mime="application/json",
                    key="main_download_btn"
                )

    with tab_json:
        st.markdown("**Load a previously saved index** to skip PDF processing.")

        # File uploader for JSON
        json_file = st.file_uploader(
            "Upload Index File",
            type=["json"],
            help="Upload a previously downloaded index JSON file",
            key="json_uploader"
        )

        if json_file is not None:
            json_file_name = json_file.name

            # Check if it's a new file (use a different key to track JSON uploads)
            json_key = f"json:{json_file_name}"
            if st.session_state.document_name != json_key:
                try:
                    index_data = load_index_from_json(json_file)
                    st.session_state.index_data = index_data
                    st.session_state.document_name = json_key
                    st.session_state.chat_history = []

                    doc_name = index_data.get("doc_name", "Unknown")
                    num_sections = len(index_data.get("structure", []))
                    st.success(f"Index loaded successfully! Document: {doc_name}, Sections: {num_sections}")
                except Exception as e:
                    st.error(f"Error loading index: {e}")

    # Chat interface
    if st.session_state.index_data:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">💬</span>
                <h3 style="margin: 0; font-weight: 600; color: #1f2937;">Ask Questions</h3>
                <span class="status-badge status-ready">● Ready</span>
            </div>
        """, unsafe_allow_html=True)

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Query input
        if query := st.chat_input("Ask a question about the document..."):
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": query})

            with st.chat_message("user"):
                st.markdown(query)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = answer_query(
                            query,
                            st.session_state.index_data,
                            st.session_state.chat_history
                        )
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error generating response: {e}"
                        st.error(error_msg)
                        import traceback
                        st.code(traceback.format_exc())
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem; opacity: 0.7;">📄</div>
                    <h3 style="margin: 0.5rem 0; font-weight: 600;">No Document Loaded</h3>
                    <p style="opacity: 0.7; margin: 0;">Upload a PDF or load a saved index to get started</p>
                </div>
            """, unsafe_allow_html=True)


def render_footer():
    """Render the footer with disclaimer."""
    st.markdown("""
        <div class="footer-text">
            <strong>🔐 Privacy First:</strong> Documents are processed locally. Only relevant text chunks are sent to the LLM API.<br>
            For sensitive documents, use a local model via Ollama.<br>
            <br>
            <a href="https://github.com/Shirish-Singh/PageIndexApp" target="_blank">GitHub</a> ·
            Built with <a href="https://github.com/VectifyAI/PageIndex" target="_blank">PageIndex</a> ·
            MIT License
        </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="PageIndex Document QA",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Clean, Professional CSS - Inspired by Notion/Linear
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global font & colors */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Main container */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* Hero header - Clean dark slate */
        .hero-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 2rem 2.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            border: 1px solid #334155;
        }
        
        .hero-header h1 {
            color: #f8fafc !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            letter-spacing: -0.02em;
        }
        
        .hero-header p {
            color: #94a3b8 !important;
            font-size: 1.05rem !important;
            margin: 0 !important;
        }
        
        .hero-badge {
            background: #0d9488;
            color: white;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
            margin-top: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Feature cards */
        .feature-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.25rem;
            height: 100%;
            transition: all 0.2s ease;
        }
        
        .feature-card:hover {
            border-color: #0d9488;
            background: #f0fdfa;
        }
        
        .feature-icon {
            font-size: 1.75rem;
            margin-bottom: 0.6rem;
        }
        
        .feature-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: #0f172a;
            margin-bottom: 0.4rem;
        }
        
        .feature-desc {
            font-size: 0.85rem;
            color: #475569;
            line-height: 1.5;
        }

        /* Chat messages */
        .stChatMessage {
            border-radius: 10px;
        }
        
        [data-testid="stChatMessageContent"] {
            font-size: 0.95rem;
            line-height: 1.6;
            color: #1e293b;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 6px;
            font-weight: 500;
            padding: 0.5rem 1rem;
            transition: all 0.15s ease;
            border: 1px solid #e2e8f0;
        }
        
        .stButton > button:hover {
            border-color: #0d9488;
            color: #0d9488;
        }
        
        .stButton > button[kind="primary"] {
            background: #0f172a;
            color: white;
            border: none;
        }
        
        .stButton > button[kind="primary"]:hover {
            background: #1e293b;
            color: white;
        }

        /* Download button */
        .stDownloadButton > button {
            background: #0d9488;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 500;
        }
        
        .stDownloadButton > button:hover {
            background: #0f766e;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            border-radius: 10px;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #f1f5f9;
            padding: 4px;
            border-radius: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            color: #475569;
        }
        
        .stTabs [aria-selected="true"] {
            background: white !important;
            color: #0f172a !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }

        /* Info/Alert boxes */
        .stAlert {
            border-radius: 8px;
            border-left-width: 3px;
        }
        
        .stSuccess {
            background: #f0fdf4;
            border-left-color: #0d9488;
        }
        
        .stInfo {
            background: #f0f9ff;
            border-left-color: #0284c7;
        }

        /* Chat input */
        [data-testid="stChatInput"] textarea {
            border-radius: 8px !important;
            border-color: #e2e8f0 !important;
        }
        
        [data-testid="stChatInput"] textarea:focus {
            border-color: #0d9488 !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            font-weight: 500;
            font-size: 0.9rem;
            color: #475569;
        }

        /* Spinner */
        .stSpinner > div {
            border-top-color: #0d9488 !important;
        }

        /* Footer */
        .footer-text {
            text-align: center;
            color: #64748b;
            font-size: 0.85rem;
            padding: 2rem 0;
            line-height: 1.8;
        }
        
        .footer-text a {
            color: #0d9488;
            text-decoration: none;
            font-weight: 500;
        }
        
        .footer-text a:hover {
            text-decoration: underline;
        }

        /* Status indicator */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        
        .status-ready {
            background: #ccfbf1;
            color: #0f766e;
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 3rem 2rem;
            background: #f8fafc;
            border: 1px dashed #cbd5e1;
            border-radius: 12px;
            margin-top: 1rem;
        }
        
        .empty-state h3 {
            color: #334155;
            font-weight: 600;
            margin-bottom: 0.3rem;
        }
        
        .empty-state p {
            color: #64748b;
            margin: 0;
        }

        /* Divider */
        hr {
            border: none;
            height: 1px;
            background: #e2e8f0;
            margin: 1.5rem 0;
        }

        /* Hide Streamlit footer but keep header for settings access */
        footer {visibility: hidden;}
        
        /* Style the header/toolbar */
        header[data-testid="stHeader"] {
            background: transparent;
        }
        
        /* ========== DARK MODE SUPPORT ========== */
        /* Streamlit dark mode detection */
        @media (prefers-color-scheme: dark) {
            .main .block-container {
                background: #0f172a;
            }
        }
        
        /* Streamlit's dark theme class */
        [data-testid="stAppViewContainer"][data-theme="dark"],
        .stApp[data-theme="dark"] {
            background: #0f172a;
        }
        
        /* Dark mode: Hero header */
        [data-testid="stAppViewContainer"][data-theme="dark"] .hero-header,
        [data-theme="dark"] .hero-header {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-color: #475569;
        }
        
        /* Dark mode: Feature cards */
        [data-testid="stAppViewContainer"][data-theme="dark"] .feature-card,
        [data-theme="dark"] .feature-card {
            background: #1e293b;
            border-color: #334155;
        }
        
        [data-testid="stAppViewContainer"][data-theme="dark"] .feature-card:hover,
        [data-theme="dark"] .feature-card:hover {
            background: #1e3a3a;
            border-color: #0d9488;
        }
        
        [data-testid="stAppViewContainer"][data-theme="dark"] .feature-title,
        [data-theme="dark"] .feature-title {
            color: #f1f5f9;
        }
        
        [data-testid="stAppViewContainer"][data-theme="dark"] .feature-desc,
        [data-theme="dark"] .feature-desc {
            color: #94a3b8;
        }
        
        /* Dark mode: Empty state */
        [data-testid="stAppViewContainer"][data-theme="dark"] .empty-state,
        [data-theme="dark"] .empty-state {
            background: #1e293b;
            border-color: #475569;
        }
        
        [data-testid="stAppViewContainer"][data-theme="dark"] .empty-state h3,
        [data-theme="dark"] .empty-state h3 {
            color: #e2e8f0;
        }
        
        [data-testid="stAppViewContainer"][data-theme="dark"] .empty-state p,
        [data-theme="dark"] .empty-state p {
            color: #94a3b8;
        }
        
        /* Dark mode: Status badge */
        [data-testid="stAppViewContainer"][data-theme="dark"] .status-ready,
        [data-theme="dark"] .status-ready {
            background: #134e4a;
            color: #5eead4;
        }
        
        /* Dark mode: Footer */
        [data-testid="stAppViewContainer"][data-theme="dark"] .footer-text,
        [data-theme="dark"] .footer-text {
            color: #64748b;
        }
        
        /* Dark mode: Sidebar */
        [data-testid="stAppViewContainer"][data-theme="dark"] [data-testid="stSidebar"],
        [data-theme="dark"] [data-testid="stSidebar"] {
            background: #1e293b;
            border-right-color: #334155;
        }
        
        /* Dark mode: Tabs */
        [data-testid="stAppViewContainer"][data-theme="dark"] .stTabs [data-baseweb="tab-list"],
        [data-theme="dark"] .stTabs [data-baseweb="tab-list"] {
            background: #1e293b;
        }
        
        [data-testid="stAppViewContainer"][data-theme="dark"] .stTabs [data-baseweb="tab"],
        [data-theme="dark"] .stTabs [data-baseweb="tab"] {
            color: #94a3b8;
        }
        
        [data-testid="stAppViewContainer"][data-theme="dark"] .stTabs [aria-selected="true"],
        [data-theme="dark"] .stTabs [aria-selected="true"] {
            background: #334155 !important;
            color: #f1f5f9 !important;
        }
        
        /* Dark mode: Chat messages */
        [data-testid="stAppViewContainer"][data-theme="dark"] [data-testid="stChatMessageContent"],
        [data-theme="dark"] [data-testid="stChatMessageContent"] {
            color: #e2e8f0;
        }
        
        /* Dark mode: Alerts */
        [data-testid="stAppViewContainer"][data-theme="dark"] .stInfo,
        [data-theme="dark"] .stInfo {
            background: #0c4a6e;
            border-left-color: #0ea5e9;
        }
        
        [data-testid="stAppViewContainer"][data-theme="dark"] .stSuccess,
        [data-theme="dark"] .stSuccess {
            background: #14532d;
            border-left-color: #22c55e;
        }
        
        /* Dark mode: Divider */
        [data-testid="stAppViewContainer"][data-theme="dark"] hr,
        [data-theme="dark"] hr {
            background: #334155;
        }
        </style>
    """, unsafe_allow_html=True)

    init_session_state()
    render_sidebar()
    render_main_content()
    render_footer()


if __name__ == "__main__":
    main()
