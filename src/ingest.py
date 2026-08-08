"""
PDF -> plain text -> overlapping chunks.
Kept deliberately simple and dependency-light (pypdf only) so it runs
reliably on Streamlit Community Cloud.
"""

from dataclasses import dataclass
from typing import List
from pypdf import PdfReader

from config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def extract_pages(pdf_path_or_file) -> List[str]:
    """Returns a list of raw text strings, one per PDF page."""
    reader = PdfReader(pdf_path_or_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages


def chunk_pages(pages: List[str], source_name: str,
                 chunk_size: int = CHUNK_SIZE_CHARS,
                 overlap: int = CHUNK_OVERLAP_CHARS) -> List[Chunk]:
    """Concatenates page text with page markers, then splits into
    overlapping character-window chunks. Overlap prevents facts from
    being severed exactly at a chunk boundary."""
    chunks: List[Chunk] = []

    for page_num, page_text in enumerate(pages, start=1):
        page_text = " ".join(page_text.split())  # collapse whitespace
        if not page_text:
            continue

        start = 0
        while start < len(page_text):
            end = min(start + chunk_size, len(page_text))
            piece = page_text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, source=source_name, page=page_num))
            if end == len(page_text):
                break
            start = end - overlap  # step back for overlap

    return chunks


def build_chunks_from_pdf(pdf_path_or_file, source_name: str = "document") -> List[Chunk]:
    pages = extract_pages(pdf_path_or_file)
    return chunk_pages(pages, source_name=source_name)
