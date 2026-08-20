"""
RAG Pipeline Implementation

Supports:
- PDF / TXT / Markdown ingestion
- Local HuggingFace embeddings
- Gemini or OpenAI LLM
- Chroma in-memory vector store
- Role filtering
- Retrieval threshold
- Source citations
- JSONL logging
- SQLite audit logging
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma

from markdown_loader import load_markdown_with_sections

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# =========================================================
# HuggingFace embeddings
# =========================================================

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None


# =========================================================
# Gemini
# =========================================================

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None


# =========================================================
# Project imports
# =========================================================

import config
from audit_sqlite import QALogRecord, SQLiteAudit, now_iso


# =========================================================
# Retrieved chunk
# =========================================================

@dataclass
class RetrievedChunk:
    doc: Document
    score: float
    idx: int


# =========================================================
# JSONL Logger
# =========================================================

class JSONLLogger:
    """Append-only JSONL logger."""

    def __init__(self, path: str = "logs/qa.jsonl"):

        self.path = path

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

    def log(self, record: dict) -> None:

        record = {
            "ts": now_iso(),
            **record,
        }

        with open(
            self.path,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# =========================================================
# RAG Pipeline
# =========================================================

class RAGPipeline:
    """RAG pipeline for document question answering."""

    def __init__(
        self,
        logger: Optional[JSONLLogger] = None,
    ):

        # -------------------------------------------------
        # Provider
        # -------------------------------------------------

        self.provider = getattr(
            config,
            "PROVIDER",
            "gemini",
        ).lower()

        self.emb_provider = getattr(
            config,
            "EMBEDDINGS_PROVIDER",
            "local",
        ).lower()

        # -------------------------------------------------
        # Embeddings
        # -------------------------------------------------

        if self.emb_provider == "local":

            if HuggingFaceEmbeddings is None:

                raise ImportError(
                    "Missing langchain-huggingface.\n\n"
                    "Run:\n"
                    "pip install -U langchain-huggingface sentence-transformers"
                )

            model_name = getattr(
                config,
                "LOCAL_EMBEDDINGS_MODEL",
                "all-MiniLM-L6-v2",
            )

            print(
                f"[RAG] Loading local embedding model: {model_name}"
            )

            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    "device": "cpu",
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                },
            )

            print(
                "[RAG] Local embeddings loaded."
            )

        elif self.emb_provider == "gemini":

            raise ValueError(
                "Gemini embeddings are not configured in this version. "
                "Use EMBEDDINGS_PROVIDER=local."
            )

        else:

            api_key = getattr(
                config,
                "OPENAI_API_KEY",
                "",
            )

            if not api_key:

                raise ValueError(
                    "OPENAI_API_KEY is required when "
                    "EMBEDDINGS_PROVIDER=openai"
                )

            self.embeddings = OpenAIEmbeddings(
                openai_api_key=api_key
            )

        # -------------------------------------------------
        # Vector store
        # -------------------------------------------------

        self.vectorstore: Optional[Chroma] = None

        # -------------------------------------------------
        # Text splitter
        # -------------------------------------------------

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(
                getattr(
                    config,
                    "CHUNK_SIZE",
                    1000,
                )
            ),
            chunk_overlap=int(
                getattr(
                    config,
                    "CHUNK_OVERLAP",
                    200,
                )
            ),
            length_function=len,
        )

        # -------------------------------------------------
        # LLM
        # -------------------------------------------------

        if self.provider == "gemini":

            if ChatGoogleGenerativeAI is None:

                raise ImportError(
                    "Gemini dependency missing.\n\n"
                    "Run:\n"
                    "pip install -U langchain-google-genai"
                )

            api_key = getattr(
                config,
                "GOOGLE_API_KEY",
                "",
            )

            if not api_key:

                raise ValueError(
                    "GOOGLE_API_KEY is required when "
                    "PROVIDER=gemini"
                )

            model = getattr(
                config,
                "MODEL_NAME",
                "gemini-3.6-flash",
            )

            print(
                f"[RAG] Gemini model: {model}"
            )

            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=float(
                    getattr(
                        config,
                        "TEMPERATURE",
                        0.3,
                    )
                ),
                google_api_key=api_key,
            )

        else:

            self.llm = ChatOpenAI(
                model_name=getattr(
                    config,
                    "MODEL_NAME",
                    "gpt-4o-mini",
                ),
                temperature=float(
                    getattr(
                        config,
                        "TEMPERATURE",
                        0.3,
                    )
                ),
                openai_api_key=getattr(
                    config,
                    "OPENAI_API_KEY",
                    "",
                ),
            )

        # -------------------------------------------------
        # Logging
        # -------------------------------------------------

        self.logger = (
            logger
            or JSONLLogger(
                getattr(
                    config,
                    "LOG_PATH",
                    "logs/qa.jsonl",
                )
            )
        )

        self.audit = SQLiteAudit(
            getattr(
                config,
                "AUDIT_DB_PATH",
                "logs/audit.db",
            )
        )

    # =====================================================
    # INGESTION
    # =====================================================

    def load_documents(
        self,
        uploaded_files,
    ) -> None:

        docs: List[Document] = []

        for uploaded_file in uploaded_files:

            suffix = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as tmp_file:

                tmp_file.write(
                    uploaded_file.getvalue()
                )

                tmp_path = tmp_file.name

            try:

                loaded = self._load_path(
                    tmp_path,
                    source_name=uploaded_file.name,
                )

                docs.extend(loaded)

            finally:

                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        self._index_documents(docs)

    # =====================================================

    def load_manifest_paths(
        self,
        paths: Iterable[str],
    ) -> None:

        docs: List[Document] = []

        for path in paths:

            path = os.path.expanduser(path)

            loaded = self._load_path(
                path,
                source_name=os.path.basename(path),
            )

            docs.extend(loaded)

        self._index_documents(docs)

    # =====================================================

    def load_manifest_docs(
        self,
        manifest_docs: Iterable[object],
    ) -> None:

        docs: List[Document] = []

        for md in manifest_docs:

            path = os.path.expanduser(
                getattr(md, "path")
            )

            source_name = os.path.basename(path)

            loaded = self._load_path(
                path,
                source_name=source_name,
            )

            for doc in loaded:

                doc.metadata = doc.metadata or {}

                doc.metadata["doc_id"] = getattr(
                    md,
                    "id",
                    None,
                )

                doc.metadata["title"] = getattr(
                    md,
                    "title",
                    None,
                )

                doc.metadata["tags"] = list(
                    getattr(
                        md,
                        "tags",
                        [],
                    )
                    or []
                )

                doc.metadata["allowed_roles"] = list(
                    getattr(
                        md,
                        "allowed_roles",
                        [],
                    )
                    or []
                )

            docs.extend(loaded)

        self._index_documents(docs)

    # =====================================================
    # LOAD FILE
    # =====================================================

    def _load_path(
        self,
        path: str,
        source_name: str,
    ) -> List[Document]:

        if path.lower().endswith(".md"):

            loaded = load_markdown_with_sections(
                path,
                source_name=source_name,
            )

            for doc in loaded:

                doc.metadata = doc.metadata or {}

                # Always use the real filename.
                doc.metadata["source"] = source_name

            return loaded

        if path.lower().endswith(".pdf"):

            loader = PyPDFLoader(path)

        elif path.lower().endswith(".txt"):

            loader = TextLoader(
                path,
                encoding="utf-8",
            )

        else:

            return []

        loaded = loader.load()

        for doc in loaded:

            doc.metadata = doc.metadata or {}

            # -------------------------------------------------
            # IMPORTANT:
            # Always use the uploaded filename instead of
            # PDF internal metadata such as about:blank.
            # -------------------------------------------------

            doc.metadata["source"] = source_name

            # -------------------------------------------------
            # Normalize page number.
            #
            # PyPDFLoader normally uses zero-based page numbers.
            # We display human-friendly one-based page numbers.
            # -------------------------------------------------

            if doc.metadata.get("page") is not None:

                try:

                    doc.metadata["page"] = (
                        int(doc.metadata["page"]) + 1
                    )

                except Exception:

                    pass

        return loaded

    # =====================================================
    # INDEX DOCUMENTS
    # =====================================================

    def _index_documents(
        self,
        documents: List[Document],
    ) -> None:

        if not documents:

            raise ValueError(
                "No supported documents found."
            )

        print(
            f"[RAG] Loaded {len(documents)} document pages."
        )

        splits = self.text_splitter.split_documents(
            documents
        )

        print(
            f"[RAG] Created {len(splits)} chunks."
        )

        per_source_counts = {}

        for doc in splits:

            source = (
                doc.metadata or {}
            ).get(
                "source",
                "unknown",
            )

            per_source_counts[source] = (
                per_source_counts.get(
                    source,
                    0,
                )
                + 1
            )

            doc.metadata["chunk"] = (
                per_source_counts[source]
            )

        # -------------------------------------------------
        # Chroma vector database
        # -------------------------------------------------

        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=None,
        )

        print(
            "[RAG] Vector database ready."
        )

    # =====================================================
    # RETRIEVAL
    # =====================================================

    def _retrieve(
        self,
        question: str,
        k: int,
        role: Optional[str] = None,
    ) -> List[RetrievedChunk]:

        if not self.vectorstore:

            raise ValueError(
                "No documents loaded. "
                "Please upload documents first."
            )

        raw_k = max(
            k * 3,
            k,
        )

        print(
            f"[RAG] Searching for: {question}"
        )

        pairs = (
            self.vectorstore
            .similarity_search_with_score(
                question,
                k=raw_k,
            )
        )

        # -------------------------------------------------
        # Convert Chroma distance to relevance.
        #
        # Chroma returns distance:
        #
        # 0 = very similar
        # larger = less similar
        #
        # Convert to a 0-1 relevance value.
        # -------------------------------------------------

        converted = []

        for doc, distance in pairs:

            try:

                distance = float(distance)

                relevance = (
                    1.0
                    / (1.0 + max(distance, 0.0))
                )

            except Exception:

                relevance = 0.0

            converted.append(
                (
                    doc,
                    relevance,
                )
            )

        pairs = converted

        # -------------------------------------------------
        # Sort by highest relevance.
        # -------------------------------------------------

        pairs.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        # -------------------------------------------------
        # Role filtering
        # -------------------------------------------------

        def allowed(
            doc: Document,
        ) -> bool:

            if (
                not role
                or role == "(all)"
            ):
                return True

            roles = (
                doc.metadata or {}
            ).get(
                "allowed_roles"
            ) or []

            if not roles:
                return True

            return role in roles

        filtered = [
            (
                doc,
                score,
            )
            for doc, score in pairs
            if allowed(doc)
        ][:k]

        # -------------------------------------------------
        # Result objects
        # -------------------------------------------------

        out: List[RetrievedChunk] = []

        for index, (
            doc,
            score,
        ) in enumerate(
            filtered,
            start=1,
        ):

            out.append(
                RetrievedChunk(
                    doc=doc,
                    score=float(score),
                    idx=index,
                )
            )

        print(
            "[RAG] Retrieved:",
            [
                round(x.score, 4)
                for x in out
            ],
        )

        return out

    # =====================================================
    # QUERY
    # =====================================================

    def query(
        self,
        question: str,
        temperature: Optional[float] = None,
        k: Optional[int] = None,
        role: Optional[str] = None,
    ):

        k = int(
            k
            or getattr(
                config,
                "K_DOCUMENTS",
                5,
            )
        )

        threshold = float(
            getattr(
                config,
                "RETRIEVAL_THRESHOLD",
                0.20,
            )
        )

        retrieved = self._retrieve(
            question,
            k=k,
            role=role,
        )

        best_score = (
            retrieved[0].score
            if retrieved
            else 0.0
        )

        print(
            f"[RAG] Best relevance: {best_score:.4f}"
        )

        # -------------------------------------------------
        # SAFETY GATE
        # -------------------------------------------------

        if (
            not retrieved
            or best_score < threshold
        ):

            answer = (
                "Not in KB yet. "
                "Please add the relevant "
                "document to the knowledge base."
            )

            sources = []

            self.logger.log(
                {
                    "question": question,
                    "best_score": best_score,
                    "k": k,
                    "status": "not_in_kb",
                    "sources": sources,
                    "answer": answer,
                }
            )

            self.audit.log(
                QALogRecord(
                    ts=now_iso(),
                    question=question,
                    status="not_in_kb",
                    best_score=best_score,
                    k=k,
                    sources=sources,
                    answer=answer,
                )
            )

            return {
                "answer": answer,
                "sources": sources,
                "retrieval": self._serialize_retrieval(
                    retrieved
                ),
            }

        # =================================================
        # BUILD CONTEXT
        # =================================================

        ctx_lines = []
        source_map = []

        for r in retrieved:

            metadata = (
                r.doc.metadata or {}
            )

            source = metadata.get(
                "source",
                "unknown",
            )

            title = metadata.get(
                "title"
            )

            section = metadata.get(
                "section_path"
            )

            chunk = metadata.get(
                "chunk"
            )

            page = metadata.get(
                "page"
            )

            label = title or source

            parts = []

            if chunk is not None:

                parts.append(
                    f"chunk {chunk}"
                )

            if page is not None:

                parts.append(
                    f"page {page}"
                )

            if section:

                parts.append(
                    f"section {section}"
                )

            ref = label

            if parts:

                ref += (
                    " ("
                    + ", ".join(parts)
                    + ")"
                )

            source_map.append(
                {
                    "id": r.idx,
                    "ref": ref,
                    "metadata": metadata,
                }
            )

            ctx_lines.append(
                f"[S{r.idx}] {ref}\n"
                f"{r.doc.page_content}"
            )

        context = "\n\n".join(
            ctx_lines
        )

        # =================================================
        # PROMPT
        # =================================================

        system = """
You are a document question-answering assistant.

IMPORTANT RULES:

1. Answer ONLY using the provided SOURCES.
2. Do not use outside knowledge.
3. Do not guess or infer missing information.
4. Every factual claim must have a citation such as [S1].
5. If the SOURCES contain the answer, answer it directly.
6. If the SOURCES do not contain the answer, say exactly:
   Not in KB yet.
7. Do not invent names, colleges, universities, companies,
   dates, addresses, skills, or any other information.
8. If the question asks for a specific person or field,
   only answer if that specific information is explicitly
   present in the SOURCES.
9. Keep the answer clear and concise.
10. At the end include a Sources section listing only the
    sources used for the answer.
"""

        user = f"""
Question:

{question}

SOURCES:

{context}

Answer using ONLY the sources above.

Remember:
- Do not guess.
- Do not use outside knowledge.
- Cite factual statements using [S1], [S2], etc.
"""

        # =================================================
        # CALL GEMINI / OPENAI
        # =================================================

        if temperature is not None:

            try:

                self.llm.temperature = float(
                    temperature
                )

            except Exception:

                pass

        print(
            "[RAG] Calling LLM..."
        )

        try:

            message = self.llm.invoke(
                [
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": user,
                    },
                ]
            )

        except Exception as exc:

            print(
                "[RAG] LLM ERROR:",
                repr(exc),
            )

            answer = (
                "Error while calling the AI model: "
                + str(exc)
            )

            return {
                "answer": answer,
                "sources": [],
                "retrieval": self._serialize_retrieval(
                    retrieved
                ),
            }

        answer_text = (
            _message_to_text(
                message
            )
            .strip()
        )

        print(
            "[RAG] LLM response received."
        )

        # =================================================
        # CITATION SAFETY
        # =================================================

        if (
            "[S"
            not in answer_text
            and "Not in KB yet"
            not in answer_text
        ):

            answer_text = (
                "Not in KB yet. "
                "The model did not provide "
                "source citations."
            )

        used_ids = sorted(
            {
                int(match.group(1))
                for match in re.finditer(
                    r"\[S(\d+)\]",
                    answer_text,
                )
            }
        )

        used_sources = [
            source
            for source in source_map
            if source["id"] in used_ids
        ]

        if not used_sources:

            used_sources = source_map

        # =================================================
        # SOURCES SECTION
        # =================================================

        if "Sources" not in answer_text:

            source_lines = [
                "",
                "",
                "Sources:",
            ]

            for source in used_sources:

                source_lines.append(
                    f"- [S{source['id']}] "
                    f"{source['ref']}"
                )

            answer_text += "\n".join(
                source_lines
            )

        # =================================================
        # LOGGING
        # =================================================

        self.logger.log(
            {
                "question": question,
                "best_score": best_score,
                "k": k,
                "status": "answered",
                "sources": [
                    source["ref"]
                    for source in used_sources
                ],
                "answer": answer_text,
            }
        )

        self.audit.log(
            QALogRecord(
                ts=now_iso(),
                question=question,
                status="answered",
                best_score=best_score,
                k=k,
                sources=[
                    source["ref"]
                    for source in used_sources
                ],
                answer=answer_text,
            )
        )

        return {
            "answer": answer_text,
            "sources": [
                source["ref"]
                for source in used_sources
            ],
            "retrieval": self._serialize_retrieval(
                retrieved
            ),
        }

    # =====================================================
    # SERIALIZE RETRIEVAL
    # =====================================================

    def _serialize_retrieval(
        self,
        retrieved: List[RetrievedChunk],
    ) -> List[dict]:

        output = []

        for r in retrieved:

            metadata = (
                r.doc.metadata or {}
            )

            output.append(
                {
                    "id": r.idx,
                    "score": r.score,
                    "score_norm": r.score,
                    "source": metadata.get(
                        "source"
                    ),
                    "chunk": metadata.get(
                        "chunk"
                    ),
                    "page": metadata.get(
                        "page"
                    ),
                }
            )

        return output


# =========================================================
# HELPERS
# =========================================================

def _message_to_text(
    message,
) -> str:

    content = getattr(
        message,
        "content",
        message,
    )

    if isinstance(
        content,
        str,
    ):

        return content

    if isinstance(
        content,
        list,
    ):

        parts: list[str] = []

        for part in content:

            if isinstance(
                part,
                str,
            ):

                parts.append(part)

            elif isinstance(
                part,
                dict,
            ):

                text = (
                    part.get("text")
                    or part.get("content")
                )

                if isinstance(
                    text,
                    str,
                ):

                    parts.append(text)

                else:

                    parts.append(
                        json.dumps(
                            part,
                            ensure_ascii=False,
                        )
                    )

            else:

                parts.append(
                    str(part)
                )

        return "\n".join(parts)

    return str(content)


def _normalize_retrieval_score(
    score: float,
) -> float:

    try:

        return max(
            0.0,
            min(
                1.0,
                float(score),
            ),
        )

    except Exception:

        return 0.0


def _extract_citation_tokens(
    text: str,
) -> List[str]:

    output = []

    for match in re.finditer(
        r"\[S\d+\]",
        text,
    ):

        token = match.group(0)

        if token not in output:

            output.append(token)

    return output