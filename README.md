# 🎯 RAG-Based Resume Matcher AI

### An Intelligent Semantic Resume-JD Alignment System

> **Built with:** LangChain · Groq (LLaMA 3.3 70B) · Qdrant · HuggingFace · Streamlit  
> **Cost to run:** $0 — 100% free stack

---

## 🧠 What Problem Does This Solve?

Traditional Applicant Tracking Systems (ATS) often rely on simple keyword matching. If a job description asks for "Python" and your resume says "Programming in Python," a basic system might miss the connection or rank it lower.

**Resume Matcher AI** uses **Retrieval-Augmented Generation (RAG)** and **Semantic Search** to understand the _meaning_ behind your experience:

```
User Resume & Job Description
     ↓
Load & Chunk documents (PDF/TXT/URL)
     ↓
Embed chunks using HuggingFace 'all-MiniLM-L6-v2'
     ↓
Index resume chunks in Qdrant (In-memory Vector DB)
     ↓
Compute Semantic Match Score (Cosine Similarity)
     ↓
Retrieve most relevant experience chunks for the JD
     ↓
Analyze alignment using LLaMA 3.3 70B
     ↓
Generate detailed Feedback, Skill Gaps & Recommendations ✅
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                      │
│         (Upload Resume + Paste JD / JD URL)         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              RAG Analytics Pipeline                 │
│                                                     │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Document │─▶│ Vector Index │─▶│ Semantic      │  │
│  │ Loader   │  │ (Qdrant Mem) │  │ Scorer        │  │
│  └──────────┘  └──────┬───────┘  └───────────────┘  │
│                        │ Relevant Chunks            │
│               ┌────────▼────────┐  ┌─────────────┐  │
│               │ LLM Analysis    │─▶│   Final     │  │
│               │ (LLaMA 3.3 70B) │  │   Report    │  │
│               └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
          │                          │
┌─────────▼─────────┐      ┌────────▼────────┐
│  HuggingFace      │      │     Groq API    │
│  (Embeddings)     │      │  LLaMA 3.3 70B  │
└───────────────────┘      └─────────────────┘
```

---

## 🛠️ Tech Stack

| Component         | Technology                     | Why                                                     |
| ----------------- | ------------------------------ | ------------------------------------------------------- |
| **Orchestration** | LangChain                      | Industry standard for LLM application workflows         |
| **LLM**           | Groq + LLaMA 3.3 70B           | Fastest free-tier LLM inference available               |
| **Embeddings**    | HuggingFace `all-MiniLM-L6-v2` | Lightweight, open-source, runs locally on CPU           |
| **Vector DB**     | Qdrant (In-Memory)             | Ultra-fast vector similarity search without cloud setup |
| **Frontend**      | Streamlit                      | Clean, interactive UI for ML applications               |
| **JD Loader**     | WebBaseLoader / Tavily         | Fetches job descriptions directly from URLs             |

---

## ✨ Features

- 📄 **Dual Format Support:** Upload resumes in PDF or TXT format.
- 💼 **URL JD Ingestion:** Paste a job link (LinkedIn, Indeed, etc.), and the AI fetches it for you.
- 📊 **Semantic Scoring:** A math-driven % match score based on vector similarity, not just keywords.
- 🧠 **Skill Gap Analysis:** Automatically identifies matched skills vs. missing requirements.
- 💡 **Actionable Coaching:** 5 specific recommendations on how to tailor the resume for that specific role.
- 🔍 **RAG Transparency:** View the exact sections of the resume the AI prioritized during analysis.
- 🆓 **Zero Cost:** Uses all free-tier APIs and local models.

---

## 📈 Semantic RAG vs Keyword Matching

| Scenario              | Keyword Search (ATS) | Semantic RAG (This App)       |
| --------------------- | -------------------- | ----------------------------- |
| Different terminology | ❌ Misses matches    | ✅ Understands context        |
| Contextual experience | ❌ Ignored           | ✅ Evaluates relevance        |
| Complex requirements  | ❌ Overwhelmed       | ✅ Breaks down JD into chunks |
| Feedback              | ❌ None              | ✅ Detailed coaching          |

---

## 🚀 Step-by-Step Local Setup

### Step 1 — Get Free API Keys

You need at least a Groq key to run the analysis:

| Service    | URL                              | Role                             |
| ---------- | -------------------------------- | -------------------------------- |
| **Groq**   | [groq.com](https://groq.com)     | LLM Inference (LLaMA 3.3 70B)    |
| **Tavily** | [tavily.com](https://tavily.com) | Fetching JD from URLs (Optional) |

---

### Step 2 — Clone & Install

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Resume_Matcher.git
cd Resume_Matcher

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> ⚠️ **Note:** On first run, the app will download the HuggingFace embedding model (~90 MB).

---

### Step 3 — Run the App

```bash
streamlit run resume_matcher.py
```

Open: **http://localhost:8501** in your browser.

---

### Step 4 — Use the App

1. **Enter your Groq API Key** in the sidebar.
2. **Upload your Resume** (PDF or TXT).
3. **Provide the Job Description** (Paste the text or use a URL).
4. **Click "🚀 Analyze Resume Match"**.
5. Receive your **Match Score**, **Skill Analysis**, and **Expert Feedback**.

---

## 📦 Project Structure

```
Resume_Matcher/
│
├── resume_matcher.py    # Main Application (LangChain + Streamlit)
├── requirements.txt     # Python Dependencies
├── README.md            # Documentation
└── .gitignore           # Python/Streamlit git ignore
```

---

## 🔬 How the RAG Pipeline Works

The system calculates matches using a sophisticated retrieval-augmented approach:

1. **Chunking**: Both Resume and JD are split into small, overlapping chunks to preserve local context.
2. **Indexing**: Only the Resume chunks are stored in the Qdrant vector database.
3. **Scoring**: For every chunk in the Job Description, we find the highest similarity score in the Resume. The average of these "best matches" forms the overall score.
4. **Context Injection**: The top $K$ most relevant resume sections are sent to the LLM (LLaMA 3.3) to provide grounded, non-hallucinated feedback.

---

## 💡 Key Concepts

**Semantic Similarity**  
Instead of looking for the word "Python," the model looks at the vector representation of the text. This allows it to understand that someone with "Deep Learning" experience likely knows "Neural Networks."

**Vector Centroid Search**  
The app computes the "center" of the job description's requirements and finds the resume sections closest to that center to ensure the LLM focuses on the most important parts of your background.

**Local Embeddings**  
By using `all-MiniLM-L6-v2` locally, your resume data remains on your machine during the embedding process, only becoming external when the final summarized context is sent to the LLM.

---

## 🙋 Author

**JAY DOBARIYA**  
AI / Python Developer

---

## 📄 License

MIT License — free to use, modify, and distribute.
