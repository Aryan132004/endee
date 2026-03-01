from __future__ import annotations
import tempfile, os
import streamlit as st
from utils.pdf_parser import file_to_chunks
from utils.embedder import embed, embed_one
from utils.vector_store import upsert_chunks, search
from utils.llm import answer
import config

st.set_page_config(page_title="Personal Knowledge Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Personal Knowledge Assistant")
st.caption("Upload your documents, then ask questions — powered by Endee + Gemini.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded = st.file_uploader("Choose PDF / TXT / MD files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploaded:
        new_files = [f for f in uploaded if f.name not in st.session_state.ingested_files]
        if new_files and st.button(f"Ingest {len(new_files)} file(s) →"):
            for uf in new_files:
                with st.spinner(f"Processing {uf.name} …"):
                    suffix = os.path.splitext(uf.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    try:
                        chunks = file_to_chunks(tmp_path)
                        for c in chunks:
                            c["source"] = uf.name
                            c["id"] = f"{uf.name}::chunk{c['chunk']}"
                        if chunks:
                            vectors = embed([c["text"] for c in chunks])
                            upsert_chunks(chunks, vectors)
                            st.session_state.ingested_files.add(uf.name)
                            st.success(f"✔ {uf.name} – {len(chunks)} chunks stored")
                        else:
                            st.warning(f"⚠ {uf.name} – no text found")
                    except Exception as e:
                        st.error(f"✖ {uf.name}: {e}")
                    finally:
                        os.unlink(tmp_path)
    if st.session_state.ingested_files:
        st.markdown("---")
        st.subheader("✅ Ingested Files")
        for fname in sorted(st.session_state.ingested_files):
            st.markdown(f"- {fname}")
    st.markdown("---")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, config.TOP_K)
    show_sources = st.checkbox("Show source chunks", value=True)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your documents …"):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base …"):
            try:
                chunks = search(embed_one(question), top_k=top_k)
                if not chunks:
                    response = "⚠️ No relevant documents found. Please upload files first."
                    st.warning(response)
                else:
                    response = answer(question, chunks)
                    st.markdown(response)
                    if show_sources:
                        with st.expander("📚 Retrieved source chunks"):
                            for i, chunk in enumerate(chunks, 1):
                                st.markdown(f"**[{i}]** `{chunk['source']}` — similarity: `{chunk['similarity']}`")
                                st.caption(chunk["text"][:300] + "…")
                                st.divider()
            except Exception as e:
                response = f"❌ Error: {e}"
                st.error(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
