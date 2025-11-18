# app.py (CPU forced KeyBERT; OpenAI v1 SDK 호환; 키워드 선택 상태 유지; Mermaid 실시간 렌더링 포함)

import os
import pickle
import faiss
import numpy as np
import streamlit as st
import fitz
import docx
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import torch
from transformers import AutoTokenizer, AutoModel
from io import BytesIO
import openai
import streamlit.components.v1 as components

st.set_page_config(page_title="Document Search & Auto Plan", layout="wide")
st.title("문서 검색 · 키워드 · AI 기획서 생성 — TEAM TechTree")
st.info("문서를 업로드하거나 ./documents 폴더에 넣어두면 자동 벡터화됩니다.")

# ---- API KEY 처리 ----
openai_key = ""
try:
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    pass
if not openai_key:
    openai_key = os.environ.get("OPENAI_API_KEY", "")

if not openai_key:
    st.warning("⚠️ OpenAI API 키가 설정되어 있지 않습니다. 기획서 생성 기능은 비활성화됩니다.")
else:
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)

# ---- CSS ----
css_file = "style.css"
if os.path.exists(css_file):
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

status_message = st.empty()
status_message.info("모델 로드 중...")

# ---- LaBSE 임베딩 모델 ----
model_name = "sentence-transformers/LaBSE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModel.from_pretrained(model_name)
hf_model.eval()
hf_model.to("cpu")

@torch.no_grad()
def get_embedding(text: str):
    if not text:
        return np.zeros(hf_model.config.hidden_size, dtype="float32")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = hf_model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype("float32")
    return emb

status_message.success("임베딩 모델 로드 완료")

# ---- KeyBERT (CPU 강제 설정) ----
torch.set_default_device("cpu")
stopwords_ko = ["은", "는", "이", "가", "의", "에", "을", "를", "와", "과", "도", "로", "으로"]
kw_model = KeyBERT(model="all-mpnet-base-v2")
kw_model.model.embedding_model.to("cpu")

# ---- FAISS DB ----
index_file = "vector_index.faiss"
data_file = "doc_data.pkl"
embedding_dim = hf_model.config.hidden_size

if os.path.exists(index_file) and os.path.exists(data_file):
    try:
        index = faiss.read_index(index_file)
        data = pickle.load(open(data_file, "rb"))
        doc_names = data.get("names", [])
        doc_paragraphs = data.get("paragraphs", [])
        doc_embeddings = data.get("embeddings", [])
        doc_keywords = data.get("keywords", [])
        status_message.success("기존 DB 로드 완료")
    except Exception:
        index = faiss.IndexFlatL2(embedding_dim)
        doc_names, doc_paragraphs, doc_embeddings, doc_keywords = [], [], [], []
else:
    index = faiss.IndexFlatL2(embedding_dim)
    doc_names, doc_paragraphs, doc_embeddings, doc_keywords = [], [], [], []
    status_message.success("새 DB 생성 완료")

# ---- 문서 읽기 ----
def read_document_paragraphs(file_path_or_file):
    paragraphs = []
    try:
        if hasattr(file_path_or_file, "read"):
            name = file_path_or_file.name
            content = file_path_or_file.read()
            if name.endswith(".pdf"):
                pdf = fitz.open(stream=content, filetype="pdf")
                for page in pdf:
                    text = page.get_text("text")
                    paragraphs += [p.strip() for p in text.split("\n") if p.strip()]
            elif name.endswith(".docx"):
                temp = BytesIO(content)
                d = docx.Document(temp)
                paragraphs += [p.text.strip() for p in d.paragraphs if p.text.strip()]
        else:
            if file_path_or_file.endswith(".pdf"):
                pdf = fitz.open(file_path_or_file)
                for page in pdf:
                    text = page.get_text("text")
                    paragraphs += [p.strip() for p in text.split("\n") if p.strip()]
            elif file_path_or_file.endswith(".docx"):
                if os.path.basename(file_path_or_file).startswith("~$"):
                    return []
                d = docx.Document(file_path_or_file)
                paragraphs += [p.text.strip() for p in d.paragraphs if p.text.strip()]
    except Exception as e:
        st.error(f"문서 읽기 실패: {e}")
    return paragraphs

# ---- 벡터화 ----
def process_file(file_path_or_file, file_name, pb=None, offset=0, total=1):
    paragraphs = read_document_paragraphs(file_path_or_file)
    for i, p in enumerate(paragraphs):
        if (file_name, i) in doc_paragraphs:
            continue
        emb = get_embedding(p)
        index.add(np.array([emb], dtype="float32"))
        doc_names.append(file_name)
        doc_paragraphs.append((file_name, i))
        doc_embeddings.append(emb)
        try:
            kw = kw_model.extract_keywords(p, keyphrase_ngram_range=(1, 2),
                                          stop_words=stopwords_ko, top_n=8)
            doc_keywords.append([k for k, _ in kw])
        except Exception:
            doc_keywords.append(p.split()[:8])
        if pb:
            pb.progress(min((offset + i + 1) / max(total, 1), 1.0))

# ---- documents 폴더 자동 처리 ----
if not os.path.exists("./documents"):
    os.makedirs("./documents")

doc_files = [f for f in os.listdir("./documents") if f.endswith((".pdf", ".docx"))]
if doc_files:
    pb = st.progress(0)
    total = sum(len(read_document_paragraphs(os.path.join('./documents', f))) for f in doc_files)
    off = 0
    for fn in doc_files:
        path = os.path.join("./documents", fn)
        process_file(path, fn, pb, off, total)
        off += len(read_document_paragraphs(path))
    pb.empty()
    st.success("documents 폴더 벡터화 완료")

# ---- 업로드 ----
with st.expander("문서 업로드 및 벡터화", expanded=True):
    uploaded = st.file_uploader("문서를 선택 (.pdf / .docx)", accept_multiple_files=True)
    if uploaded:
        pb = st.progress(0)
        total = 0
        paths = []
        for f in uploaded:
            name = f.name
            if not name.endswith((".pdf", ".docx")):
                continue
            temp = os.path.join("./documents", name)
            open(temp, "wb").write(f.getbuffer())
            paths.append((name, temp))
            total += len(read_document_paragraphs(temp))
        off = 0
        for name, temp in paths:
            process_file(temp, name, pb, off, total)
            off += len(read_document_paragraphs(temp))
        pb.empty()
        st.success("업로드 문서 벡터화 완료")

# ---- DB 저장 ----
def save_db():
    faiss.write_index(index, index_file)
    pickle.dump({
        "names": doc_names,
        "paragraphs": doc_paragraphs,
        "embeddings": doc_embeddings,
        "keywords": doc_keywords
    }, open(data_file, "wb"))

if st.button("DB 저장"):
    save_db()
    st.success("DB 저장 완료")

# ---- 세션 상태 초기화 ----
if "kw_selection" not in st.session_state:
    st.session_state["kw_selection"] = []

if "last_search_result" not in st.session_state:
    st.session_state["last_search_result"] = []

# ---- 검색 ----
with st.expander("문서 검색", expanded=True):
    query = st.text_input("검색어 입력")
    top_k = st.slider("상위 결과 수", 1, 10, 5)
    if st.button("검색 실행"):
        if not doc_embeddings:
            st.warning("DB가 비어 있습니다.")
        else:
            q_emb = get_embedding(query).reshape(1, -1).astype("float32")
            k = min(top_k, len(doc_embeddings))
            dist, idxs = index.search(q_emb, k)
            st.subheader(f"검색 결과 ({k}개)")
            all_kw = []
            results = []
            for rank, idx in enumerate(idxs[0]):
                sim = cosine_similarity([q_emb[0]], [doc_embeddings[idx]])[0][0]
                fn, pidx = doc_paragraphs[idx]
                kws = doc_keywords[idx]
                all_kw += kws
                results.append({
                    "file": fn,
                    "paragraph_idx": pidx,
                    "similarity": sim,
                    "keywords": kws
                })
                st.markdown(f"**{rank+1}. {fn} - 문단 {pidx}**")
                st.markdown(f"유사도: {sim:.4f} | 키워드: {', '.join(kws)}")

            st.session_state["last_search_result"] = results
            st.session_state["kw_selection"] = list(dict.fromkeys(all_kw))[:6]  # 기본 선택 6개

    # ---- 키워드 선택 UI ----
    if st.session_state["last_search_result"]:
        st.markdown("**검색 키워드 선택**")
        st.session_state["kw_selection"] = st.multiselect(
            "기획서에 넣을 키워드 선택",
            options=list(dict.fromkeys([kw for r in st.session_state["last_search_result"] for kw in r["keywords"]])) ,
            default=st.session_state["kw_selection"]
        )

# ---- 기획서 생성 ----
def generate_project_plan(keywords, notes=""):
    if not openai_key:
        raise RuntimeError("API 키 없음")
    kw = ", ".join(keywords)
    prompt = f"""
당신은 소프트웨어 기획 전문가입니다. 아래 키워드 기반으로 '문서 검색 및 키워드 추출 시스템' 기획서를 작성하세요.
키워드: {kw}
추가요청: {notes}

[PLAN] 항목 포함:
- 프로젝트 개요
- 문제 정의 & 목표
- 핵심 기능(우선순위)
- 아키텍처(간단한 이유 포함)
- 개발 로드맵
- 기대 효과

[MERMAID] flowchart 코드 1개 포함
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1200
    )
    text = res.choices[0].message.content
    if "[MERMAID]" in text:
        plan, mermaid = text.split("[MERMAID]")
        plan = plan.replace("[PLAN]", "").strip()
        mermaid = mermaid.strip()
    else:
        plan = text
        mermaid = "flowchart LR\nA[Upload] --> B[Embedding] --> C[FAISS] --> D[Search] --> E[Plan]"
    return plan, mermaid

with st.expander("AI 자동 기획서 생성", expanded=True):
    default_text = ", ".join(st.session_state.get("kw_selection", []))
    kw_input = st.text_area("사용할 키워드", value=default_text)
    notes = st.text_area("추가 요청")
    if st.button("기획서 생성"):
        if not openai_key:
            st.error("API 키가 없어 기능을 사용할 수 없습니다.")
        elif not kw_input.strip():
            st.warning("키워드를 입력하세요.")
        else:
            with st.spinner("생성 중..."):
                kws = [k.strip() for k in kw_input.split(",") if k.strip()]
                plan, mermaid = generate_project_plan(kws, notes)
                st.success("완료")
                
                # 계획서 텍스트
                st.markdown(plan)
                
                # Mermaid 렌더링
                components.html(f"""
                <div class="mermaid">
                {mermaid}
                </div>
                <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true }});
                </script>
                """, height=500)
                
                # 다운로드 버튼
                def create_docx_bytes(plan, mermaid):
                    doc = Document()
                    doc.add_heading("AI Project Plan", 0)
                    for line in plan.split("\n"):
                        doc.add_paragraph(line)
                    doc.add_heading("Mermaid Code", 1)
                    for line in mermaid.split("\n"):
                        doc.add_paragraph(line)
                    bio = BytesIO()
                    doc.save(bio)
                    bio.seek(0)
                    return bio

                st.download_button(
                    "DOCX 다운로드",
                    data=create_docx_bytes(plan, mermaid),
                    file_name="project_plan.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                st.download_button(
                    "Mermaid(.mmd) 다운로드",
                    data=mermaid,
                    file_name="diagram.mmd",
                    mime="text/plain"
                )

st.caption("DB 저장을 눌러 벡터 DB를 보존하세요.")
