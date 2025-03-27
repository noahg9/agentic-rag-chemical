import streamlit as st
from meta_agent import run_meta_agent
import plotly.graph_objects as go
import json
from config import RESOURCES_PATH
from pathlib import Path
import re
from datetime import datetime

def get_default_questions():
    return [
        "Which personal protective equipment do I need to use during Hypophosphorous Acid Addition?",
        "Describe in detail the steps that I need to do for the Sulfated Analysis by HPLC.",
        "Provide me the raw materials to be used for the synthesis of Alkylbenzen Sulfonic Acid and their SAP Code.",
        "When synthesizing Alkylbenzen sulfonic acid, which should be the setpoint for the sulfur trioxide when doing the sulfonation?",
        "Which range of humidity values are acceptable for the Alkylbenzen Sulfonic Acid?",
        "Describe the hazard classification of the AN-84 product.",
        "Describe the amidation reaction for the production of AN-84.",
        "Describe, in detail, the operational method for the production of Texapon S80.",
        "Which of the following products require water as raw material: AS-42, DETON PK-45, Texapon S80, TOMPERLAN OCD, Alkylbenzen Sulfonic Acid and AN-84?",
        "Can I produce AS-42 at the R003 reactor?",
        "Is there wastewater generation for AS-42 production?",
        "Which are the raw materials used for the production of AS-42? How much of each raw material should be used for the production of 1 Ton of AS-42?",
        "How much of each raw material should be used for the production of 3 Tons of AS-42?"
    ]


def init_session_state():
    defaults = {
        'history': [],
        'show_metadata': True,
        'compact_view': False,
        'auto_expand_docs': False,
        'theme_color': '#00ff00',
        'language': 'en',
        'font_size': 'medium',
        'confidence_threshold': 80  # Moved here as default
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def create_confidence_gauge(confidence):
    color = st.session_state.theme_color
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50],
                 'color': f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)"},
                {'range': [50, 75],
                 'color': f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.4)"},
                {'range': [75, 100],
                 'color': f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.6)"}
            ],
        }
    ))
    fig.update_layout(height=200)
    return fig


import streamlit as st
import re
from datetime import datetime
from pathlib import Path
from config import RESOURCES_PATH
from extract_metadata import extract_metadata_from_pdf, extract_metadata_from_docx

def sanitize_filename(name):
    """Sanitize filenames for safe storage and retrieval."""
    return re.sub(r'[\\/*?:"<>| ]+', "_", str(name)).strip()

def parse_pdf_date(pdf_date):
    """Parses dates found in PDFs to a standard format."""
    try:
        match = re.match(
            r'D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})([+-Z])(\d{2})?\'?(\d{2})?',
            pdf_date
        )
        if match:
            year, month, day, hour, minute, sec = map(int, match.groups()[:6])
            return datetime(year, month, day, hour, minute, sec).strftime("%Y-%m-%d %H:%M:%S")
        return pdf_date
    except (ValueError, TypeError, AttributeError) as e:
        st.error(f"Date parsing error: {str(e)}")
        return pdf_date

def find_matching_file(base_name, ext):
    """Find an exact match for a file in RESOURCES_PATH by extension."""
    for f in Path(RESOURCES_PATH).glob(f"*{ext}"):
        if sanitize_filename(f.stem).lower() == base_name.lower():
            return f
    return None

def find_likely_file(base_name, ext):
    """
    As a fallback, return the first file whose sanitized stem contains the base_name.
    """
    candidates = []
    for f in Path(RESOURCES_PATH).glob(f"*{ext}"):
        if base_name.lower() in sanitize_filename(f.stem).lower():
            candidates.append(f)
    if candidates:
        return candidates[0]
    return None

### Display Functions for PDFs and DOCX

def display_pdf_document(metadata, page_content, doc_title, i):
    """Display a PDF document (Work Instruction)."""
    sanitized_name = sanitize_filename(doc_title)
    # Try exact match first, then fallback to likely match
    doc_file = find_matching_file(sanitized_name, ".pdf")
    if not doc_file:
        doc_file = find_likely_file(sanitized_name, ".pdf")

    with st.expander(f"📋 WI {doc_title} - Source {i}", expanded=st.session_state.auto_expand_docs):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("### 📑 Document Metadata")
            if doc_file and doc_file.exists():
                with open(doc_file, "rb") as f:
                    st.download_button(
                        label="📥 Download PDF",
                        data=f,
                        file_name=doc_file.name,
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"download_button_{i}"
                    )
            else:
                st.warning(f"Original document not found for: {doc_title}")

            st.markdown(f"""
            **Version:** {metadata.get("Version", "N/A")}  
            **Document Date:** {parse_pdf_date(metadata.get("Date", "N/A"))}  
            **Author:** {metadata.get("Author", metadata.get("author", "N/A"))}  
            **Safety Measures:** {', '.join(metadata.get("Safety Measures", [])) if metadata.get("Safety Measures") else 'None'}
            """)
        with col2:
            st.markdown("### 📝 Relevant Content")
            st.markdown(f"```\n{page_content[:1000]}...\n```")
        st.markdown("---")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"""
            **File Format:** {metadata.get("format", "N/A")}  
            **Creator:** {metadata.get("creator", "N/A")}  
            **Producer:** {metadata.get("producer", "N/A")}
            """)
        with cols[1]:
            mod_date = parse_pdf_date(metadata.get("modDate", "N/A")) or "N/A"
            doc_id = metadata.get("_id", f"doc_{i}")
            st.markdown(f"""
            **Collection:** {metadata.get("_collection_name", "N/A")}  
            **Modified:** {mod_date}  
            **Document ID:** `{doc_id}`
            """)

def display_docx_document(metadata, page_content, doc_title, i):
    """Display a DOCX document (SOP)."""
    sanitized_name = sanitize_filename(doc_title)
    doc_file = find_matching_file(sanitized_name, ".docx")
    if not doc_file:
        doc_file = find_likely_file(sanitized_name, ".docx")

    with st.expander(f"📑 SOP {doc_title} - Source {i}", expanded=st.session_state.auto_expand_docs):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("### 📑 Document Metadata")
            if doc_file and doc_file.exists():
                with open(doc_file, "rb") as f:
                    st.download_button(
                        label="📥 Download DOCX",
                        data=f,
                        file_name=doc_file.name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        key=f"download_button_{i}"
                    )
            else:
                st.warning(f"Original document not found for: {doc_title}")
            st.markdown(f"""
            **Review:** {metadata.get("Review", "N/A")}  
            **Date:** {parse_pdf_date(metadata.get("Date", "N/A"))}
            """)
        with col2:
            st.markdown("### 📝 Relevant Content")
            st.markdown(f"```\n{page_content[:1000]}...\n```")
        st.markdown("---")
        st.markdown("### 📄 Additional Information")
        st.markdown(f"**File Name:** {doc_title}")

### Main Display Function

def display_source_documents(docs):
    if not docs:
        st.warning("No source documents found")
        return

    for i, doc in enumerate(docs, 1):
        try:
            # Handle both dictionary-based and object-based document formats
            if isinstance(doc, dict):
                page_content = doc.get("page_content", "")
                # Assume the remaining keys constitute metadata
                metadata = {k: v for k, v in doc.items() if k != "page_content"}
            else:
                metadata = getattr(doc, "metadata", {})
                page_content = getattr(doc, "page_content", "")

            # Determine document title
            doc_title = metadata.get("Document Name") or metadata.get("title") or "Unknown Document"
            sanitized_name = sanitize_filename(doc_title)

            # Determine file extension from metadata if possible
            if doc_title.lower().startswith("work instruction"):
                ext = ".pdf"
            elif doc_title.lower().startswith("sop") or ("Review" in metadata and metadata.get("Review")):
                ext = ".docx"
            else:
                fmt = str(metadata.get("format", "")).lower()
                if "pdf" in fmt:
                    ext = ".pdf"
                elif "doc" in fmt:
                    ext = ".docx"
                else:
                    ext = ""

            # Check if a file_path exists in the document object; if not, try to build one
            file_path = doc.get("file_path", "").strip()
            if not file_path and ext:
                file_path = str(Path(RESOURCES_PATH) / f"{sanitized_name}{ext}")

            # Route document based on file extension.
            if file_path.endswith(".pdf"):
                display_pdf_document(metadata, page_content, doc_title, i)
            elif file_path.endswith(".docx"):
                display_docx_document(metadata, page_content, doc_title, i)
            else:
                st.warning(f"⚠️ Unsupported file type or missing file path for: {doc_title}")
        except Exception as e:
            st.error(f"Error displaying document {i}: {str(e)}")
            st.write("Raw metadata for debugging:")
            st.json(metadata if "metadata" in locals() else {})




def main():
    init_session_state()

    st.set_page_config(
        page_title="Chemical Operations Assistant",
        page_icon="⚗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.title("⚗️ Settings")

        # Confidence threshold setting
        st.markdown("### 🎯 Answer Settings")
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0,
            max_value=100,
            value=st.session_state.confidence_threshold,
            help="Set minimum confidence level for answers"
        )
        st.session_state.confidence_threshold = confidence

        st.divider()

        # Enhanced UI Settings section with tooltips
        st.markdown("### 🎨 UI Settings")
        tabs = st.tabs(["Theme", "Display", "Export"])

        with tabs[0]:
            st.color_picker("Theme Color", value=st.session_state.theme_color, key="theme_color")
            st.select_slider(
                "Font Size",
                options=["small", "medium", "large"],
                value=st.session_state.font_size,
                key="font_size"
            )

        with tabs[1]:
            st.toggle("Show Metadata", key="show_metadata", help="Display detailed document information")
            st.toggle("Compact View", key="compact_view", help="Show metadata in compact JSON format")
            st.toggle("Auto-expand Sources", key="auto_expand_docs", help="Automatically expand source documents")

        with tabs[2]:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Export History", use_container_width=True):
                    history_data = json.dumps(st.session_state.history, default=str)
                    st.download_button(
                        "Download JSON",
                        history_data,
                        file_name=f"chem_ops_history_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
            with col2:
                if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                    if st.session_state.history:
                        st.session_state.history = []
                        st.rerun()

    # Main content with custom styling
    st.markdown(f"""
        <style>
            .main .block-container {{ font-size: {{'small': '0.9em', 'medium': '1em', 'large': '1.1em'}}[st.session_state.font_size]; }}
            .stButton>button {{ background-color: {st.session_state.theme_color}; }}
        </style>
    """, unsafe_allow_html=True)

    st.title("⚗️ Chemical Operations Assistant")
    st.markdown("---")

    # Query section with tabs
    query_tab, history_tab = st.tabs(["📝 Query", "📜 History"])

    with query_tab:
        query_type = st.radio(
            "Select Query Type",
            ["Default Questions", "Custom Query"],
            horizontal=True
        )

        if query_type == "Default Questions":
            query = st.selectbox(
                "Choose a benchmark question:",
                options=get_default_questions(),
                index=None,
                placeholder="Select one of our benchmark questions..."
            )
        else:
            query = st.text_area("Enter your custom query:", placeholder="Type your question here...")

        col1, col2 = st.columns([6, 1])
        with col2:
            submit = st.button("Submit", type="primary", use_container_width=True)

        if submit and query:
            with st.spinner("Processing your query..."):
                result = run_meta_agent(query)
                st.session_state.history.append({
                    "timestamp": datetime.now(),
                    "query": query,
                    "result": result
                })

                st.markdown("### 📌 Answer")
                st.markdown(result["combined_answer"])


                st.plotly_chart(create_confidence_gauge(result["confidence"]))

                st.markdown("### 📚 Source Documents")
                display_source_documents(result["source_documents"])

    with history_tab:
        if st.session_state.history:
            for item in reversed(st.session_state.history):
                with st.expander(f"🔍 {item['query']} - {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                    st.markdown(f"**Answer:** {item['result']['combined_answer']}")
                    st.progress(item['result']['confidence'] / 100)


if __name__ == "__main__":
    main()