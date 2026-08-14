"""
RAG layer: ingest provided datasets/reference artifacts and retrieve
relevant chunks for a product BEFORE falling back to live web search.

This directly answers the brief's "supporting datasets and reference
artifacts may also be provided" line, and gives you a real, demoable
RAG story: "for provided/known data we retrieve locally and instantly;
for unknown parts we fall back to live discovery."

Implementation note: uses TF-IDF + cosine similarity (scikit-learn)
rather than a downloaded embedding model. This is a deliberate choice
for hackathon reliability -- no large model download over conference
wifi, no external embedding API cost, and it's fast to set up. It is
a legitimate, explainable retrieval method (you can literally show
judges which keywords matched). If you have reliable internet on demo
day and want to upgrade to semantic embeddings (sentence-transformers
or an API embedding model), swap out `_vectorize()` -- everything else
in this file stays the same.
"""
import os
import glob
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models import SourceHit

CHUNK_SIZE = 800  # characters per chunk
DATASET_DIR = os.getenv("DATASET_DIR", "./datasets")


class LocalRAGStore:
    """
    Holds ingested chunks in memory and answers similarity queries.
    Rebuilt fresh on server start -- fine for a hackathon's dataset size.
    """

    def __init__(self):
        self.chunks: list[str] = []
        self.chunk_sources: list[str] = []  # which file each chunk came from
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.ready = False

    def ingest_directory(self, directory: str = DATASET_DIR) -> int:
        """
        Reads every .txt/.csv/.pdf file in `directory`, splits into
        chunks, and builds the TF-IDF index. Returns number of chunks
        ingested. Safe to call with an empty/missing directory --
        the RAG store just stays empty and discover.py falls back to
        web search for everything.
        """
        self.chunks = []
        self.chunk_sources = []

        if not os.path.isdir(directory):
            self.ready = False
            return 0

        for filepath in glob.glob(os.path.join(directory, "**", "*"), recursive=True):
            if os.path.isdir(filepath):
                continue
            if "output" in os.path.basename(filepath).lower():
                continue
            text = self._read_file(filepath)
            if not text:
                continue
            for i in range(0, len(text), CHUNK_SIZE):
                chunk = text[i:i + CHUNK_SIZE].strip()
                if len(chunk) > 50:  # skip near-empty tail chunks
                    self.chunks.append(chunk)
                    self.chunk_sources.append(os.path.basename(filepath))

        if not self.chunks:
            self.ready = False
            return 0

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)
        self.ready = True
        return len(self.chunks)

    def query(self, text: str, top_k: int = 3, min_score: float = 0.1, part_number: str = "") -> list[SourceHit]:
        """
        Returns the top_k most similar chunks as SourceHit objects
        (origin="rag"), or an empty list if the store isn't ready or
        nothing scores above min_score.

        IMPORTANT: if a part_number is provided, a chunk is only
        trusted if that exact part number actually appears in its text.
        Similarity score alone is too loose a bar -- generic product
        documents can score above min_score against an unrelated part's
        query just by sharing common words (units, spec labels, etc),
        which would silently attribute one product's data to a totally
        different product. Requiring the part number to literally be
        present is a hard, cheap correctness check on top of the
        similarity ranking.
        """
        if not self.ready:
            return []

        query_vec = self.vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        normalized_pn = _normalize_part_number(part_number) if part_number else None

        hits = []
        for idx, score in ranked:
            if score < min_score:
                continue
            chunk_text = self.chunks[idx]
            if normalized_pn and normalized_pn not in _normalize_part_number(chunk_text):
                # Similarity matched on generic wording, but the actual
                # part number isn't in this chunk -- this is a false
                # positive, not real evidence about this product. Skip it.
                continue
            hits.append(
                SourceHit(
                    url=f"dataset://{self.chunk_sources[idx]}",
                    title=f"Provided dataset: {self.chunk_sources[idx]}",
                    snippet=chunk_text[:200],
                    raw_text=chunk_text,
                    origin="rag",
                )
            )
        return hits

    @staticmethod
    def _read_file(filepath: str) -> str:
        ext = filepath.lower().rsplit(".", 1)[-1] if "." in filepath else ""
        try:
            if ext == "pdf":
                text_parts = []
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                return "\n".join(text_parts)
            elif ext in ("txt", "csv", "md", "json"):
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        except Exception as e:
            print(f"[rag] failed to read {filepath}: {e}")
        return ""


# Single shared instance used across the app -- built once at startup.
rag_store = LocalRAGStore()


def _normalize_part_number(text: str) -> str:
    """Strips spaces/dashes/case so '6ES7 214-1AG40-0XB0' and '6es7214-1ag40-0xb0' match."""
    return "".join(ch for ch in text.lower() if ch.isalnum())
