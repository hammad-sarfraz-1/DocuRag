import io
from typing import List

from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.config import Config


# ---------------------------------------------------------------------------
# OCR (optional — disabled if EasyOCR not available or config says so)
# ---------------------------------------------------------------------------
_ocr_reader = None


def _init_ocr():
    global _ocr_reader
    if _ocr_reader is None and Config.USE_OCR:
        try:
            import easyocr

            _ocr_reader = easyocr.Reader(Config.OCR_LANG, gpu=False)
        except ImportError:
            pass  # OCR not available
    return _ocr_reader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF. Falls back to OCR when standard extraction
    yields little or no text."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    text = text.strip()

    if Config.USE_OCR and (not text or len(text) < 50):
        ocr = _init_ocr()
        if ocr is not None:
            try:
                from pdf2image import convert_from_bytes
                import numpy as np

                images = convert_from_bytes(file_bytes)
                ocr_lines = []
                for img in images:
                    img_np = np.array(img)
                    result = ocr.readtext(img_np, detail=0)
                    ocr_lines.extend(result)
                text = "\n".join(ocr_lines)
            except ImportError:
                pass  # pdf2image or numpy not installed
    return text


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Route a file to the correct parser based on its extension."""
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif filename.lower().endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif filename.lower().endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks for embedding storage."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def build_chunk_metadata(chunks: List[str], source: str) -> List[dict]:
    """Build per-chunk metadata dicts (used for source citation)."""
    return [
        {
            "source": source,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]
