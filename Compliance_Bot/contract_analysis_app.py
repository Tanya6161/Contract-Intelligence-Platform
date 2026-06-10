import os
import re
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Contract Intelligence",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Page background */
.stApp {
    background-color: #0f1117;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #252d3d;
}

/* Hide default Streamlit header decoration */
[data-testid="stHeader"] { background: transparent; }

/* ── Hero banner ── */
.hero-wrap {
    background: linear-gradient(135deg, #1a2340 0%, #0d1220 100%);
    border: 1px solid #252d3d;
    border-radius: 12px;
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 1.5rem;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6ee7b7;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f0f4ff;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #8899bb;
    max-width: 520px;
}

/* ── Cards ── */
.card {
    background: #161b27;
    border: 1px solid #252d3d;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* ── Risk / Negotiation items ── */
.risk-item {
    background: #1e1212;
    border-left: 3px solid #f87171;
    border-radius: 6px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
}
.risk-item .label { color: #f87171; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.risk-item .clause-text { color: #fca5a5; font-weight: 600; font-size: 0.92rem; margin-bottom: 0.2rem; }
.risk-item .reason-text { color: #9ca3af; font-size: 0.87rem; }

.neg-item {
    background: #0f1e18;
    border-left: 3px solid #34d399;
    border-radius: 6px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
}
.neg-item .label { color: #34d399; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.neg-item .issue-text { color: #6ee7b7; font-weight: 600; font-size: 0.92rem; margin-bottom: 0.2rem; }
.neg-item .suggestion-text { color: #9ca3af; font-size: 0.87rem; }

.section-item {
    background: #141b2e;
    border-left: 3px solid #818cf8;
    border-radius: 6px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
}
.section-item .sec-name { color: #a5b4fc; font-weight: 600; font-size: 0.92rem; margin-bottom: 0.2rem; }
.section-item .sec-desc { color: #9ca3af; font-size: 0.87rem; }

/* ── Summary box ── */
.summary-box {
    background: #141b2e;
    border: 1px solid #252d3d;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    color: #c7d2f0;
    font-size: 0.95rem;
    line-height: 1.65;
}

/* ── Status chips ── */
.chip {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.chip-green { background: #052e1d; color: #34d399; border: 1px solid #065f46; }
.chip-blue  { background: #0d1f3c; color: #60a5fa; border: 1px solid #1e3a5f; }
.chip-red   { background: #2d0d0d; color: #f87171; border: 1px solid #5f1e1e; }

/* ── Agent pills in sidebar ── */
.agent-pill {
    background: #0d1220;
    border: 1px solid #252d3d;
    border-radius: 8px;
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
}
.agent-pill .icon { font-size: 1.1rem; margin-top: 1px; }
.agent-pill .agent-name { color: #c7d2f0; font-size: 0.84rem; font-weight: 600; }
.agent-pill .agent-role { color: #6b7280; font-size: 0.76rem; }

/* ── Progress steps ── */
.step-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.55rem 0.8rem;
    border-radius: 7px;
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
}
.step-active { background: #0d1f3c; color: #93c5fd; }
.step-done   { background: #052e1d; color: #6ee7b7; }
.step-pending{ background: #161b27; color: #4b5563; }

/* ── Button override ── */
.stButton > button {
    background: linear-gradient(90deg, #4f46e5 0%, #6366f1 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; }

/* ── Sidebar section label ── */
.sidebar-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4b5563;
    margin: 1rem 0 0.5rem;
}

/* ── Tab styling ── */
[data-baseweb="tab"] { background: transparent !important; }
[data-baseweb="tab-list"] { background: #161b27 !important; border-radius: 8px; border: 1px solid #252d3d; }
[data-baseweb="tab"][aria-selected="true"] { background: #1e2d4a !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #161b27;
    border: 1.5px dashed #2d3a50;
    border-radius: 10px;
}

/* ── Metric ── */
[data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #252d3d;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────
load_dotenv("my.env", override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found")
    st.stop()


# ─────────────────────────────────────────────
# LLM + VECTOR STORE
# ─────────────────────────────────────────────
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)

VECTOR_DB = None


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────
@tool("Ingest Contract PDF")
def ingest_contract_pdf_tool(pdf_path: str) -> str:
    """Load contract PDF, chunk text, generate embeddings, and store in Chroma vector DB."""
    global VECTOR_DB
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    texts = splitter.split_text(full_text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    VECTOR_DB = Chroma.from_texts(texts=texts, embedding=embeddings)
    return f"Contract ingested: {len(texts)} chunks created."


@tool("Retrieve Contract Context")
def retrieve_contract_context(query: str) -> str:
    """Retrieve relevant contract clauses using semantic similarity search."""
    if VECTOR_DB is None:
        return "ERROR: Contract not yet ingested."
    docs = VECTOR_DB.similarity_search(query, k=5)
    return "\n\n".join([doc.page_content[:500] for doc in docs])


@tool("Detect Key Contract Sections")
def detect_key_sections(text: str) -> str:
    """Rule-based detection of important contract sections."""
    keywords = {
        "Termination":    ["terminate", "termination", "expiry"],
        "Payment":        ["payment", "fee", "invoice", "compensation"],
        "Liability":      ["liability", "indemnify", "damages"],
        "Confidentiality":["confidential", "non-disclosure"],
        "Governing Law":  ["jurisdiction", "governing law", "court"]
    }
    detected = [s for s, keys in keywords.items() if any(k in text.lower() for k in keys)]
    return ", ".join(detected) if detected else "General Clauses"


# ─────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────
ingest_agent = Agent(
    role="Contract Ingestion Agent",
    goal="Prepare contract documents for downstream analysis",
    backstory="You specialize in loading PDFs, chunking text, and creating searchable embeddings for legal analysis.",
    tools=[ingest_contract_pdf_tool],
    llm=llm,
    verbose=False,
    allow_delegation=False
)

retrieval_agent = Agent(
    role="Contract Clause Retrieval Agent",
    goal="Retrieve and classify important contract clauses",
    backstory="You excel at semantic search and keyword detection to find payment, liability, termination, and key legal sections.",
    tools=[retrieve_contract_context, detect_key_sections],
    llm=llm,
    verbose=False,
    allow_delegation=False
)

analysis_agent = Agent(
    role="Contract Negotiation Analyst",
    goal="Summarize contracts and suggest negotiation improvements",
    backstory="You are a legal contract analyst specializing in identifying risks, key clauses, and negotiation opportunities.",
    llm=llm,
    verbose=False,
    allow_delegation=False
)


# ─────────────────────────────────────────────
# TASKS
# ─────────────────────────────────────────────
ingest_task = Task(
    description="Ingest the contract PDF located at {pdf_path}",
    expected_output="Confirmation message with the number of chunks created.",
    agent=ingest_agent,
)

retrieve_task = Task(
    description="Retrieve important clauses and identify key contract sections.\nQuery focus: {query}",
    expected_output="Relevant contract clauses plus detected key sections as comma-separated values.",
    agent=retrieval_agent,
    context=[ingest_task]
)

analysis_task = Task(
    description=(
        "Using retrieved contract clauses, analyze and return STRICTLY valid JSON.\n\n"
        "Exact structure required:\n"
        "{\n"
        '  "contract_summary": "High-level summary",\n'
        '  "key_sections": [{"section_name": "...", "description": "..."}],\n'
        '  "risky_clauses": [{"clause": "...", "risk_reason": "..."}],\n'
        '  "negotiation_points": [{"issue": "...", "suggested_change": "..."}]\n'
        "}\n\n"
        "Rules: No extra keys. No text outside JSON. Output must be valid JSON only."
    ),
    expected_output="Valid JSON with keys: contract_summary, key_sections, risky_clauses, negotiation_points.",
    agent=analysis_agent,
    context=[retrieve_task]
)

crew = Crew(
    agents=[ingest_agent, retrieval_agent, analysis_agent],
    tasks=[ingest_task, retrieve_task, analysis_task],
    process=Process.sequential
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def extract_json_from_text(text: str):
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{.*\}',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, re.DOTALL):
            m = match[0] if isinstance(match, tuple) else match
            try:
                return json.loads(m.strip())
            except json.JSONDecodeError:
                continue
    # brace-count fallback
    lines = text.split('\n')
    start = next((i for i, l in enumerate(lines) if l.strip().startswith('{')), -1)
    if start != -1:
        count = 0
        for i in range(start, len(lines)):
            count += lines[i].count('{') - lines[i].count('}')
            if count == 0:
                try:
                    return json.loads('\n'.join(lines[start:i+1]))
                except json.JSONDecodeError:
                    break
    return None


def fallback_json():
    return {
        "contract_summary": "Could not parse AI output. Review raw output below.",
        "key_sections": [{"section_name": "Parse Error", "description": "Check raw JSON tab."}],
        "risky_clauses": [{"clause": "N/A", "risk_reason": "Parsing failed."}],
        "negotiation_points": [{"issue": "N/A", "suggested_change": "Re-run analysis."}]
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.75rem 0 0.5rem'>
        <span style='font-family: Space Grotesk, sans-serif; font-size: 1.05rem;
            font-weight: 700; color: #c7d2f0;'>📑 Contract Intelligence</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color: #252d3d; margin: 0.5rem 0 0.75rem'>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-label'>How it works</div>", unsafe_allow_html=True)
    steps = [
        ("1", "Upload your contract PDF"),
        ("2", "AI agents ingest & chunk the document"),
        ("3", "Clauses retrieved via semantic search"),
        ("4", "Risks & negotiation points surfaced"),
    ]
    for num, desc in steps:
        st.markdown(f"""
        <div style='display:flex; gap:0.6rem; align-items:center; padding:0.35rem 0;'>
            <span style='background:#1e2d4a; color:#60a5fa; font-size:0.72rem;
                font-weight:700; border-radius:50%; width:1.35rem; height:1.35rem;
                display:inline-flex; align-items:center; justify-content:center;
                flex-shrink:0;'>{num}</span>
            <span style='color:#8899bb; font-size:0.83rem;'>{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-label' style='margin-top:1.25rem'>Active agents</div>",
                unsafe_allow_html=True)
    agents_info = [
        ("📥", "Ingestion Agent", "PDF loader · chunker · embedder"),
        ("🔍", "Retrieval Agent", "Semantic search · section detection"),
        ("⚖️", "Analyst Agent",  "Risk scoring · negotiation strategy"),
    ]
    for icon, name, role in agents_info:
        st.markdown(f"""
        <div class='agent-pill'>
            <span class='icon'>{icon}</span>
            <div>
                <div class='agent-name'>{name}</div>
                <div class='agent-role'>{role}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #252d3d; margin: 1.1rem 0 0.75rem'>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0d1220; border:1px solid #252d3d; border-radius:8px;
        padding:0.8rem 1rem; font-size:0.81rem; color:#6b7280; line-height:1.55;'>
        ⚠️ AI-assisted analysis only. Consult a qualified legal professional before acting on results.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-wrap'>
    <div class='hero-eyebrow'>AI-Powered · CrewAI + Groq</div>
    <div class='hero-title'>Contract Analysis Assistant</div>
    <div class='hero-sub'>Upload any contract PDF and get structured clause analysis,
        risk flags, and negotiation points — in seconds.</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UPLOAD + SETTINGS ROW
# ─────────────────────────────────────────────
col_up, col_cfg = st.columns([3, 2], gap="large")

with col_up:
    st.markdown("<div style='color:#8899bb; font-size:0.82rem; font-weight:600; "
                "letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;'>"
                "Contract PDF</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop or click to browse",
        type=["pdf"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        size_kb = uploaded_file.size / 1024
        st.markdown(f"""
        <div style='display:flex; gap:0.5rem; align-items:center; margin-top:0.5rem;'>
            <span class='chip chip-green'>✓ Loaded</span>
            <span style='color:#6b7280; font-size:0.83rem;'>
                {uploaded_file.name} &nbsp;·&nbsp; {size_kb:.1f} KB
            </span>
        </div>
        """, unsafe_allow_html=True)

with col_cfg:
    st.markdown("<div style='color:#8899bb; font-size:0.82rem; font-weight:600; "
                "letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;'>"
                "Configuration</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card' style='margin-bottom:0'>
        <div style='display:flex; justify-content:space-between; margin-bottom:0.6rem;'>
            <span style='color:#6b7280; font-size:0.83rem;'>Model</span>
            <span class='chip chip-blue'>Groq Llama 3.3 70B</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:0.6rem;'>
            <span style='color:#6b7280; font-size:0.83rem;'>Output tokens</span>
            <span style='color:#c7d2f0; font-size:0.83rem;'>700</span>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span style='color:#6b7280; font-size:0.83rem;'>Search query</span>
            <span style='color:#c7d2f0; font-size:0.83rem;'>Clauses · risks · obligations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ANALYSE BUTTON
# ─────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run = st.button("🚀 Analyse Contract", use_container_width=True)

# ─────────────────────────────────────────────
# ANALYSIS WORKFLOW
# ─────────────────────────────────────────────
if run:
    if not uploaded_file:
        st.warning("⚠️ Please upload a contract PDF before running the analysis.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # --- Progress display ---
        progress_slot = st.empty()

        def render_steps(active: int):
            labels = [
                ("📥", "Ingesting document"),
                ("🔍", "Retrieving clauses"),
                ("⚖️", "Analysing & scoring"),
            ]
            html = "<div style='margin:1rem 0;'>"
            for i, (icon, label) in enumerate(labels):
                if i < active:
                    cls, badge = "step-done", "✓"
                elif i == active:
                    cls, badge = "step-active", "…"
                else:
                    cls, badge = "step-pending", str(i + 1)
                html += f"""
                <div class='step-row {cls}'>
                    <span style='width:1.4rem; height:1.4rem; border-radius:50%;
                        background:rgba(255,255,255,0.06); display:inline-flex;
                        align-items:center; justify-content:center; font-size:0.78rem;
                        font-weight:700; flex-shrink:0;'>{badge}</span>
                    <span>{icon} {label}</span>
                </div>"""
            html += "</div>"
            return html

        progress_slot.markdown(render_steps(0), unsafe_allow_html=True)

        try:
            progress_slot.markdown(render_steps(1), unsafe_allow_html=True)
            with st.spinner("Running agents — this may take 30–60 seconds…"):
                progress_slot.markdown(render_steps(2), unsafe_allow_html=True)
                result = crew.kickoff(inputs={
                    "pdf_path": pdf_path,
                    "query": "important clauses, risks, and obligations"
                })

            os.unlink(pdf_path)
            progress_slot.markdown(render_steps(3), unsafe_allow_html=True)

            st.markdown("""
            <div style='background:#052e1d; border:1px solid #065f46; border-radius:8px;
                padding:0.85rem 1.1rem; color:#6ee7b7; font-weight:600; margin:0.75rem 0;'>
                ✅ Analysis complete
            </div>
            """, unsafe_allow_html=True)

            # --- Parse JSON ---
            json_result = extract_json_from_text(result.raw)
            if json_result is None:
                json_result = fallback_json()
                st.warning("⚠️ Could not parse AI output as JSON — showing default structure.")

            # --- Results tabs ---
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            tab_sum, tab_sec, tab_risk, tab_neg, tab_raw = st.tabs([
                "📝 Summary",
                "🔑 Key Sections",
                "⚠️ Risks",
                "🤝 Negotiation",
                "{ } Raw JSON",
            ])

            with tab_sum:
                st.markdown(
                    f"<div class='summary-box'>{json_result.get('contract_summary', 'No summary available.')}</div>",
                    unsafe_allow_html=True
                )

            with tab_sec:
                sections = json_result.get("key_sections", [])
                if sections:
                    for s in sections:
                        st.markdown(f"""
                        <div class='section-item'>
                            <div class='sec-name'>§ {s.get('section_name', '—')}</div>
                            <div class='sec-desc'>{s.get('description', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No key sections identified.")

            with tab_risk:
                risks = json_result.get("risky_clauses", [])
                if risks:
                    st.markdown(
                        f"<span class='chip chip-red'>⚠️ {len(risks)} risk{'s' if len(risks) != 1 else ''} found</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                    for r in risks:
                        st.markdown(f"""
                        <div class='risk-item'>
                            <div class='label'>Risk</div>
                            <div class='clause-text'>{r.get('clause', '—')}</div>
                            <div class='reason-text'>{r.get('risk_reason', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No significant risks identified.")

            with tab_neg:
                points = json_result.get("negotiation_points", [])
                if points:
                    for p in points:
                        st.markdown(f"""
                        <div class='neg-item'>
                            <div class='label'>Negotiation point</div>
                            <div class='issue-text'>{p.get('issue', '—')}</div>
                            <div class='suggestion-text'>💡 {p.get('suggested_change', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No specific negotiation points surfaced.")

            with tab_raw:
                st.json(json_result)

        except Exception as e:
            progress_slot.empty()
            st.error(f"❌ Analysis failed: {e}")
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr style='border-color:#252d3d; margin-top:2.5rem'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#374151; font-size:0.8rem; padding-bottom:1rem;'>
    Contract Intelligence · CrewAI + Groq Llama 3.3 70B + Streamlit
</div>
""", unsafe_allow_html=True)
