from dataclasses import dataclass

from app.models import Document, DocumentChunk


@dataclass
class ChunkedDocument:
    title: str
    content: str
    chunks: list[str]


def chunk_document_text(content: str, chunk_size: int = 360) -> list[str]:
    normalized = content.replace("\r", "")
    paragraphs = [part.strip() for part in normalized.split("\n") if part.strip()]
    if not paragraphs:
        return [content.strip()] if content.strip() else []

    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = paragraph if not buffer else f"{buffer}\n{paragraph}"
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer)
            buffer = ""

        if len(paragraph) <= chunk_size:
            buffer = paragraph
            continue

        for idx in range(0, len(paragraph), chunk_size):
            part = paragraph[idx: idx + chunk_size].strip()
            if part:
                chunks.append(part)

    if buffer:
        chunks.append(buffer)

    return chunks


def build_document(title: str | None, content: str) -> ChunkedDocument:
    cleaned = content.strip()
    chunks = chunk_document_text(cleaned)
    final_title = (title or "reference-note").strip() or "reference-note"
    return ChunkedDocument(title=final_title, content=cleaned, chunks=chunks)


def persist_document(session_id: str, title: str | None, content: str) -> tuple[Document, list[DocumentChunk]]:
    prepared = build_document(title, content)
    document = Document(session_id=session_id, title=prepared.title, source_type="pasted", raw_content=prepared.content)
    chunks = [
        DocumentChunk(
            session_id=session_id,
            document_id=document.id,
            chunk_index=index,
            content=chunk,
            meta={"length": len(chunk)},
        )
        for index, chunk in enumerate(prepared.chunks, start=1)
    ]
    return document, chunks
