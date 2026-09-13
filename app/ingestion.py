"""
Multimodal ingestion pipeline.

Supported inputs: PDF, PPTX, DOCX, images (printed text via OCR),
and handwritten notes (via TrOCR handwriting model).

Each file is turned into a list of (text, page_or_slide_label) tuples,
which are then chunked and handed to rag.py for embedding + indexing.
"""
import os
from typing import List, Tuple

from pypdf import PdfReader
from pptx import Presentation
from docx import Document as DocxDocument
from PIL import Image
import pytesseract

# Handwriting recognition (TrOCR) — loaded lazily since it's a large model.
_trocr_processor = None
_trocr_model = None


def _load_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_processor is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        _trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        _trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return _trocr_processor, _trocr_model


def extract_pdf(path: str) -> List[Tuple[str, str]]:
    reader = PdfReader(path)
    out = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            out.append((text, f"Page {i}"))
    return out


def extract_pptx(path: str) -> List[Tuple[str, str]]:
    prs = Presentation(path)
    out = []
    for i, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                texts.append(shape.text_frame.text)
        joined = "\n".join(t for t in texts if t.strip())
        if joined.strip():
            out.append((joined, f"Slide {i}"))
    return out


def extract_docx(path: str) -> List[Tuple[str, str]]:
    doc = DocxDocument(path)
    # DOCX has no reliable page boundaries pre-render, so we bucket by
    # every ~40 paragraphs as an approximate "section" reference.
    out = []
    bucket, section_no, count = [], 1, 0
    for para in doc.paragraphs:
        if para.text.strip():
            bucket.append(para.text)
            count += 1
        if count >= 40:
            out.append(("\n".join(bucket), f"Section {section_no}"))
            bucket, count = [], count
            section_no += 1
            count = 0
    if bucket:
        out.append(("\n".join(bucket), f"Section {section_no}"))
    return out


def extract_printed_image(path: str) -> List[Tuple[str, str]]:
    """OCR for printed/scanned text (e.g. a photographed textbook page)."""
    text = pytesseract.image_to_string(Image.open(path))
    return [(text, os.path.basename(path))] if text.strip() else []


def extract_handwritten_image(path: str) -> List[Tuple[str, str]]:
    """Handwriting recognition using TrOCR (offline, HuggingFace model)."""
    processor, model = _load_trocr()
    image = Image.open(path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return [(text, os.path.basename(path))] if text.strip() else []


def extract_document(path: str, file_type: str, is_handwritten: bool = False) -> List[Tuple[str, str]]:
    """
    Dispatch to the right extractor based on file_type.
    file_type: 'pdf' | 'pptx' | 'docx' | 'image'
    is_handwritten only applies to file_type == 'image'.
    """
    if file_type == "pdf":
        return extract_pdf(path)
    if file_type == "pptx":
        return extract_pptx(path)
    if file_type == "docx":
        return extract_docx(path)
    if file_type == "image":
        return extract_handwritten_image(path) if is_handwritten else extract_printed_image(path)
    raise ValueError(f"Unsupported file_type: {file_type}")


def chunk_text(text: str, max_words: int = 180, overlap: int = 30) -> List[str]:
    """Simple sliding-window word chunker — good enough for a baseline RAG system."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks
