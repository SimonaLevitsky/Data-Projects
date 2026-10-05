"""
RAG Part 1 — Policy Vectorization
=================================
Reads the credit underwriting policy (credit_policy.txt), splits it into
section-aware chunks, embeds them with OpenAI, and persists a local FAISS index
in ./policy_index/.

Run standalone:
    python build_vector_db.py            # build (or rebuild) the index
    python build_vector_db.py --query "What is the maximum DTI?"   # build + smoke test

The app (app.py) imports `load_or_build_index()` so the index is created
automatically on first launch if it does not exist yet.
"""

import os
import re
import sys
import json
import argparse

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(BASE_DIR, "credit_policy.txt")
INDEX_DIR = os.path.join(BASE_DIR, "policy_index")
EMBEDDING_MODEL = "text-embedding-3-small"

# A sub-section longer than this is split further (with overlap) so that each
# chunk stays well inside the embedding model's "sweet spot".
MAX_CHUNK_CHARS = 1000
CHUNK_OVERLAP = 120

SECTION_RE = re.compile(r"^SECTION\s+(\d+):\s*(.+?)\s*$", re.MULTILINE)
SUBSECTION_RE = re.compile(r"^(\d+\.\d+)\s+(.+?)\s*$", re.MULTILINE)
RULE_LINE_RE = re.compile(r"^═+\s*$", re.MULTILINE)


# ─────────────────────────────────────────────
# API KEY
# ─────────────────────────────────────────────

def get_api_key(verbose: bool = True) -> str:
    """
    Same sources Streamlit's st.secrets uses, in order:
      1. OPENAI_API_KEY environment variable
      2. <project>/.streamlit/secrets.toml
      3. ~/.streamlit/secrets.toml   (user-level file — this is usually where it lives when
                                      `streamlit run` works but a plain `python` script does not)
    """
    key = os.getenv("OPENAI_API_KEY")
    if key:
        if verbose:
            print("API key source: environment variable")
        return key
    candidates = [
        os.path.join(BASE_DIR, ".streamlit", "secrets.toml"),
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
    ]
    for secrets_path in candidates:
        if os.path.exists(secrets_path):
            try:
                import tomllib
                with open(secrets_path, "rb") as f:
                    key = tomllib.load(f).get("OPENAI_API_KEY")
                if key:
                    if verbose:
                        print(f"API key source: {secrets_path}")
                    return key
            except Exception as e:
                print(f"Could not read {secrets_path}: {e}")
    raise RuntimeError(
        "OPENAI_API_KEY not found. Looked in: the environment variable, "
        + ", ".join(candidates)
        + ". If the Streamlit app runs elsewhere (e.g. Streamlit Cloud), the key lives in that platform's secrets."
    )


# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

def load_policy_text(path: str = POLICY_PATH) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Policy document not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_policy(text: str) -> list[Document]:
    """
    Two-level, structure-aware chunking:
      1. Split on 'SECTION N: TITLE' headers.
      2. Inside each section, split on 'N.M Subtitle' sub-headers.
      3. Any sub-section longer than MAX_CHUNK_CHARS is split recursively
         with overlap.
    Every chunk is prefixed with its section/sub-section path so the embedding
    carries the context, and the same path is stored as metadata for citations.
    """
    text = RULE_LINE_RE.sub("", text)  # drop the ═══ decoration lines

    docs: list[Document] = []
    section_matches = list(SECTION_RE.finditer(text))
    if not section_matches:
        raise ValueError("No 'SECTION N:' headers found — check credit_policy.txt format.")

    # Document preamble (title / version) — kept as its own small chunk
    preamble = text[: section_matches[0].start()].strip()
    if preamble:
        docs.append(Document(
            page_content=preamble,
            metadata={"section": "0", "section_title": "Document Header",
                      "subsection": "0.0", "subsection_title": "Header", "source": "credit_policy.txt"},
        ))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n- ", "\n", ". ", " "],
    )

    for i, sec in enumerate(section_matches):
        sec_num, sec_title = sec.group(1), sec.group(2).strip()
        sec_end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(text)
        body = text[sec.end():sec_end].strip()

        sub_matches = list(SUBSECTION_RE.finditer(body))
        if not sub_matches:
            sub_blocks = [(f"{sec_num}.0", sec_title, body)]
        else:
            sub_blocks = []
            for j, sub in enumerate(sub_matches):
                sub_end = sub_matches[j + 1].start() if j + 1 < len(sub_matches) else len(body)
                sub_blocks.append((sub.group(1), sub.group(2).strip(), body[sub.end():sub_end].strip()))

        for sub_num, sub_title, sub_body in sub_blocks:
            if not sub_body:
                continue
            header = f"Section {sec_num}: {sec_title} — {sub_num} {sub_title}"
            pieces = splitter.split_text(sub_body) if len(sub_body) > MAX_CHUNK_CHARS else [sub_body]
            for k, piece in enumerate(pieces):
                docs.append(Document(
                    page_content=f"{header}\n{piece}",
                    metadata={
                        "section": sec_num,
                        "section_title": sec_title,
                        "subsection": sub_num,
                        "subsection_title": sub_title,
                        "part": k + 1,
                        "source": "credit_policy.txt",
                    },
                ))
    return docs


# ─────────────────────────────────────────────
# BUILD / LOAD
# ─────────────────────────────────────────────

def build_index(api_key: str, verbose: bool = True) -> FAISS:
    text = load_policy_text()
    docs = chunk_policy(text)
    if verbose:
        print(f"Loaded policy: {len(text):,} chars → {len(docs)} chunks")
        for d in docs:
            m = d.metadata
            print(f"  [{m['subsection']:>4}] {m['subsection_title'][:45]:<45} {len(d.page_content):>5} chars")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key, timeout=60, max_retries=2)
    store = FAISS.from_documents(docs, embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    store.save_local(INDEX_DIR)
    if verbose:
        print(f"Saved FAISS index ({store.index.ntotal} vectors, dim={store.index.d}) → {INDEX_DIR}")
    return store


def load_or_build_index(api_key: str) -> FAISS:
    """Load the persisted index; build it from credit_policy.txt if missing."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key, timeout=60, max_retries=2)
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return build_index(api_key, verbose=False)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the credit policy FAISS index.")
    parser.add_argument("--query", help="Optional smoke-test query to run after building.")
    parser.add_argument("-k", type=int, default=3, help="Top-k results for the smoke test.")
    args = parser.parse_args()

    import time
    t0 = time.time()
    store = build_index(get_api_key())
    print(f"Build time: {time.time() - t0:.1f}s")

    if args.query:
        print(f"\nSmoke test — query: {args.query!r}")
        for doc, score in store.similarity_search_with_score(args.query, k=args.k):
            m = doc.metadata
            print(f"\n--- score={score:.4f} | §{m['subsection']} {m['subsection_title']}")
            print(doc.page_content[:400])
