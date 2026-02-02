"""
RAG-Based Resume Matching System
Stack: Groq (LLaMA 3.3 70B) · HuggingFace Embeddings · Qdrant (in-memory) · Tavily · Streamlit
100% Free
"""

import os
import re
import tempfile
import streamlit as st
import numpy as np
from typing import List
from typing_extensions import TypedDict

# ── LangChain ────────────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ── Qdrant (in-memory — no cloud needed) ─────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Matcher AI",
    page_icon="🎯",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.match-score-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px; border-radius: 16px; text-align: center; color: white;
    margin: 10px 0;
}
.match-score-num {
    font-size: 64px; font-weight: 900; margin: 0;
}
.match-label {
    font-size: 18px; opacity: 0.9; margin-top: 6px;
}
.skill-tag-match {
    display:inline-block; background:#d4edda; color:#155724;
    padding:4px 12px; border-radius:20px; margin:3px; font-size:14px;
}
.skill-tag-miss {
    display:inline-block; background:#f8d7da; color:#721c24;
    padding:4px 12px; border-radius:20px; margin:3px; font-size:14px;
}
.section-card {
    background:#f8f9fa; padding:20px; border-radius:12px;
    border-left:4px solid #667eea; margin:10px 0; color:#1a1a1a;
}
</style>
""", unsafe_allow_html=True)

st.title("🎯 RAG-Based Resume Matching System")
st.caption("Semantic resume–JD alignment · Skill gap analysis · LLM-powered insights")

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Keys")
    groq_api_key   = st.text_input("Groq API Key", type="password",
                                    help="Free at groq.com")
    tavily_api_key = st.text_input("Tavily API Key (optional)", type="password",
                                    help="Only needed if loading JD from URL · Free at tavily.com")

    st.divider()
    st.header("⚙️ Settings")
    chunk_size    = st.slider("Chunk size",    200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk overlap",   0,  200,  80, 20)
    top_k         = st.slider("Top-K retrieval", 1,   10,   5,  1)

    st.divider()
    st.markdown("**Free Stack Used:**")
    st.markdown("🟠 Groq — LLaMA 3.3 70B LLM")
    st.markdown("🤗 HuggingFace — all-MiniLM-L6-v2")
    st.markdown("🔵 Qdrant — in-memory vector store")
    st.markdown("🟢 Tavily — JD URL fetching (optional)")

# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING MODEL (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model (one-time ~90MB download)…")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf_or_txt(uploaded_file, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    if suffix == ".pdf":
        docs = PyPDFLoader(tmp_path).load()
    else:
        docs = TextLoader(tmp_path).load()
    return "\n".join(d.page_content for d in docs)


def load_url(url: str) -> str:
    docs = WebBaseLoader(url).load()
    return "\n".join(d.page_content for d in docs)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = splitter.create_documents([text])
    return [d.page_content for d in docs]


def embed_texts(texts: List[str], embeddings) -> np.ndarray:
    vecs = embeddings.embed_documents(texts)
    return np.array(vecs)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def build_qdrant_index(chunks: List[str], chunk_vecs: np.ndarray,
                        collection: str) -> QdrantClient:
    """Store resume chunks in an in-memory Qdrant collection."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=chunk_vecs.shape[1], distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=i, vector=chunk_vecs[i].tolist(), payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=collection, points=points)
    return client


def retrieve_top_k(client: QdrantClient, collection: str,
                   query_vec: np.ndarray, k: int) -> List[str]:
    response = client.query_points(
        collection_name=collection,
        query=query_vec.tolist(),
        limit=k,
        with_payload=True,
    )
    return [r.payload["text"] for r in response.points]


def compute_overall_score(resume_vecs: np.ndarray, jd_vecs: np.ndarray) -> float:
    """
    Average of max-cosine-similarity for each JD chunk against all resume chunks.
    This measures how well the resume covers each part of the JD.
    """
    scores = []
    for jd_vec in jd_vecs:
        sims = [cosine_similarity(jd_vec, r_vec) for r_vec in resume_vecs]
        scores.append(max(sims))
    return round(np.mean(scores) * 100, 1)


def score_to_label(score: float) -> tuple:
    if score >= 80:
        return "🟢 Excellent Match", "#28a745"
    elif score >= 65:
        return "🟡 Good Match", "#ffc107"
    elif score >= 50:
        return "🟠 Moderate Match", "#fd7e14"
    else:
        return "🔴 Low Match", "#dc3545"


# ─────────────────────────────────────────────────────────────────────────────
#  LLM ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_analysis(resume_text: str, jd_text: str,
                     retrieved_resume_chunks: List[str],
                     match_score: float, llm) -> dict:
    """Run 4 focused LLM calls for structured analysis."""

    context = "\n---\n".join(retrieved_resume_chunks)

    # ── 1. Matched skills ────────────────────────────────────────────────
    matched_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a resume analyst. Extract ONLY a comma-separated list of skills, "
         "tools, or qualifications that appear in BOTH the resume and the job description. "
         "Return only the comma-separated list, nothing else. Max 15 items."),
        ("human",
         "RESUME EXCERPT:\n{resume}\n\nJOB DESCRIPTION:\n{jd}"),
    ])

    # ── 2. Missing skills ────────────────────────────────────────────────
    missing_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a resume analyst. Extract ONLY a comma-separated list of skills, "
         "tools, or qualifications that the job description REQUIRES but are MISSING "
         "or NOT MENTIONED in the resume. Return only the comma-separated list, nothing else. "
         "Max 15 items."),
        ("human",
         "RESUME EXCERPT:\n{resume}\n\nJOB DESCRIPTION:\n{jd}"),
    ])

    # ── 3. Overall analysis ──────────────────────────────────────────────
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert technical recruiter. Analyze how well the resume matches "
         "the job description. Write 3-4 clear paragraphs covering: "
         "(1) Overall fit based on the {score}% match score "
         "(2) Strongest alignment areas "
         "(3) Key gaps and concerns "
         "(4) Honest hiring recommendation. "
         "Be specific and professional."),
        ("human",
         "RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{jd}\n\nMatch Score: {score}%"),
    ])

    # ── 4. Recommendations ───────────────────────────────────────────────
    reco_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a career coach. Based on the resume and job description, give 5 "
         "specific, actionable recommendations the candidate should follow to improve "
         "their chances. Number each recommendation. Be concrete and practical."),
        ("human",
         "RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{jd}"),
    ])

    parser = StrOutputParser()

    matched_skills_raw = (matched_prompt | llm | parser).invoke({
        "resume": context, "jd": jd_text[:3000]
    })
    missing_skills_raw = (missing_prompt | llm | parser).invoke({
        "resume": context, "jd": jd_text[:3000]
    })
    analysis = (analysis_prompt | llm | parser).invoke({
        "resume": resume_text[:3000], "jd": jd_text[:3000], "score": match_score
    })
    recommendations = (reco_prompt | llm | parser).invoke({
        "resume": resume_text[:3000], "jd": jd_text[:3000]
    })

    def parse_csv(text: str) -> List[str]:
        items = [i.strip().strip("•-*").strip()
                 for i in re.split(r"[,\n]", text) if i.strip()]
        return [i for i in items if 2 < len(i) < 60]

    return {
        "matched_skills":  parse_csv(matched_skills_raw),
        "missing_skills":  parse_csv(missing_skills_raw),
        "analysis":        analysis,
        "recommendations": recommendations,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN UI — INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Resume")
    resume_type = st.radio("Resume format", ["PDF", "TXT"], horizontal=True,
                            key="resume_type")
    resume_file = st.file_uploader(
        "Upload your resume",
        type=["pdf"] if resume_type == "PDF" else ["txt"],
        key="resume_upload",
    )

with col2:
    st.subheader("💼 Job Description")
    jd_source = st.radio("JD source", ["Paste Text", "Enter URL"], horizontal=True,
                          key="jd_source")
    if jd_source == "Paste Text":
        jd_text_input = st.text_area(
            "Paste the Job Description here",
            height=250,
            placeholder="We are looking for a Python developer with experience in...",
        )
        jd_url_input = ""
    else:
        jd_url_input = st.text_input("Job Description URL",
                                      placeholder="https://jobs.example.com/python-dev")
        jd_text_input = ""

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYZE BUTTON
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
analyze_btn = st.button("🚀 Analyze Resume Match", use_container_width=True,
                         type="primary")

if analyze_btn:

    # ── Validation ────────────────────────────────────────────────────────
    errors = []
    if not groq_api_key:
        errors.append("Groq API Key is required")
    if not resume_file:
        errors.append("Please upload your resume")
    if not jd_text_input.strip() and not jd_url_input.strip():
        errors.append("Please provide a Job Description (paste text or enter URL)")
    if jd_url_input.strip() and not tavily_api_key:
        errors.append("Tavily API Key is required to load JD from URL")
    if errors:
        for e in errors:
            st.error(f"❌ {e}")
        st.stop()

    progress = st.progress(0, text="Starting analysis…")

    try:
        embeddings = get_embeddings()

        # ── Step 1: Load documents ────────────────────────────────────────
        progress.progress(10, text="📄 Loading resume…")
        suffix = ".pdf" if resume_type == "PDF" else ".txt"
        resume_text = load_pdf_or_txt(resume_file, suffix)

        progress.progress(20, text="💼 Loading job description…")
        if jd_url_input.strip():
            os.environ["TAVILY_API_KEY"] = tavily_api_key
            jd_text = load_url(jd_url_input.strip())
        else:
            jd_text = jd_text_input.strip()

        if len(resume_text.strip()) < 50:
            st.error("Resume appears to be empty or unreadable. Please try a different file.")
            st.stop()
        if len(jd_text.strip()) < 50:
            st.error("Job description appears to be too short. Please provide more detail.")
            st.stop()

        # ── Step 2: Chunk ─────────────────────────────────────────────────
        progress.progress(30, text="✂️ Chunking documents…")
        resume_chunks = chunk_text(resume_text, chunk_size, chunk_overlap)
        jd_chunks     = chunk_text(jd_text,     chunk_size, chunk_overlap)

        # ── Step 3: Embed ─────────────────────────────────────────────────
        progress.progress(45, text="🔢 Generating embeddings…")
        resume_vecs = embed_texts(resume_chunks, embeddings)
        jd_vecs     = embed_texts(jd_chunks,     embeddings)

        # ── Step 4: Index resume in Qdrant ────────────────────────────────
        progress.progress(55, text="🔵 Building vector index…")
        qdrant_client = build_qdrant_index(resume_chunks, resume_vecs, "resume")

        # ── Step 5: Compute match score ───────────────────────────────────
        progress.progress(65, text="📊 Computing match score…")
        match_score = compute_overall_score(resume_vecs, jd_vecs)

        # ── Step 6: Retrieve best matching resume chunks ──────────────────
        progress.progress(72, text="🔍 Retrieving relevant sections…")
        jd_centroid = np.mean(jd_vecs, axis=0)
        top_resume_chunks = retrieve_top_k(qdrant_client, "resume",
                                            jd_centroid, min(top_k, len(resume_chunks)))

        # ── Step 7: LLM analysis ──────────────────────────────────────────
        progress.progress(80, text="🧠 Running LLM analysis (this takes ~15 sec)…")
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key,
                       temperature=0.1)
        results = run_llm_analysis(
            resume_text, jd_text, top_resume_chunks, match_score, llm
        )

        progress.progress(100, text="✅ Analysis complete!")

        # ─────────────────────────────────────────────────────────────────
        #  RESULTS
        # ─────────────────────────────────────────────────────────────────
        st.divider()
        st.header("📊 Match Results")

        label, color = score_to_label(match_score)

        # ── Score display ─────────────────────────────────────────────────
        r1, r2, r3 = st.columns([1, 1, 1])

        with r1:
            st.markdown(f"""
            <div class="match-score-box">
                <p class="match-score-num">{match_score}%</p>
                <p class="match-label">Overall Match Score</p>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.metric("✅ Matched Skills",
                       len(results["matched_skills"]), delta="found in both")

        with r3:
            st.metric("❌ Missing Skills",
                       len(results["missing_skills"]), delta="not in resume",
                       delta_color="inverse")

        # ── Match label ───────────────────────────────────────────────────
        st.markdown(f"### Verdict: {label}")
        progress_val = min(int(match_score), 100)
        st.progress(progress_val / 100)

        st.divider()

        # ── Skills columns ────────────────────────────────────────────────
        sc1, sc2 = st.columns(2)

        with sc1:
            st.subheader("✅ Matched Skills")
            if results["matched_skills"]:
                tags = " ".join(
                    f'<span class="skill-tag-match">{s}</span>'
                    for s in results["matched_skills"]
                )
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.info("No strong skill matches found.")

        with sc2:
            st.subheader("❌ Missing / Gap Skills")
            if results["missing_skills"]:
                tags = " ".join(
                    f'<span class="skill-tag-miss">{s}</span>'
                    for s in results["missing_skills"]
                )
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.success("No significant skill gaps found!")

        st.divider()

        # ── LLM Analysis ─────────────────────────────────────────────────
        st.subheader("🧠 Detailed Analysis")
        st.markdown(f'<div class="section-card">{results["analysis"]}</div>',
                    unsafe_allow_html=True)

        st.divider()

        # ── Recommendations ───────────────────────────────────────────────
        st.subheader("💡 Recommendations to Improve Your Match")
        st.markdown(results["recommendations"])

        st.divider()

        # ── Retrieved resume sections ─────────────────────────────────────
        with st.expander("🔍 Most Relevant Resume Sections (retrieved via RAG)"):
            for i, chunk in enumerate(top_resume_chunks, 1):
                st.markdown(f"**Chunk {i}**")
                st.text(chunk)
                st.divider()

        # ── Raw texts ─────────────────────────────────────────────────────
        with st.expander("📄 View Extracted Texts"):
            t1, t2 = st.tabs(["Resume", "Job Description"])
            with t1:
                st.text_area("Resume Text", resume_text, height=300,
                              disabled=True)
            with t2:
                st.text_area("Job Description Text", jd_text, height=300,
                              disabled=True)

        # ── Pipeline summary ──────────────────────────────────────────────
        with st.expander("⚙️ Pipeline Summary"):
            st.markdown(f"""
| Step | Details |
|------|---------|
| Resume chunks | {len(resume_chunks)} |
| JD chunks | {len(jd_chunks)} |
| Embedding model | `all-MiniLM-L6-v2` (384-dim) |
| Vector DB | Qdrant in-memory |
| LLM | LLaMA 3.3 70B via Groq |
| Similarity metric | Cosine similarity |
| Match score method | Mean of max-cosine per JD chunk |
| Top-K retrieved | {len(top_resume_chunks)} resume chunks |
            """)

    except Exception as e:
        st.error(f"❌ Error during analysis: {e}")
        st.info("Common fixes: Check your API keys · Make sure resume has readable text · Try a shorter JD")
        progress.empty()
