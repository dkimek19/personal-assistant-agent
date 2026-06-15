"""Tests for the document I/O tool (assistant.tools.documents).

Covers:
- read_pdf: missing file -> FileNotFoundError, multi-page text extraction,
  pages with no extractable text, empty PDF, corrupt PDF -> DocumentError
- read_docx: missing file -> FileNotFoundError, paragraph extraction
  (round-trip with create_docx), corrupt DOCX -> DocumentError
- create_docx: input validation (empty/whitespace string or list), return
  value, parent-directory creation, string content split into paragraphs by
  newline, list content preserved in order, write failure -> DocumentError
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from assistant.tools.documents import (
    DocumentError,
    create_docx,
    read_docx,
    read_pdf,
)


# ---------------------------------------------------------------------------
# Tests for read_pdf
# ---------------------------------------------------------------------------


class TestReadPdf:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_pdf(tmp_path / "missing.pdf")

    def test_missing_file_error_mentions_path(self, tmp_path):
        missing = tmp_path / "missing.pdf"

        with pytest.raises(FileNotFoundError) as exc_info:
            read_pdf(missing)

        assert str(missing) in str(exc_info.value)

    @patch("assistant.tools.documents.PdfReader")
    def test_extracts_text_from_all_pages(self, mock_reader_cls, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        page1 = MagicMock()
        page1.extract_text.return_value = "Page one text"
        page2 = MagicMock()
        page2.extract_text.return_value = "Page two text"
        mock_reader_cls.return_value.pages = [page1, page2]

        text = read_pdf(pdf_path)

        assert text == "Page one text\n\nPage two text"

    @patch("assistant.tools.documents.PdfReader")
    def test_page_with_no_extractable_text_becomes_empty_string(self, mock_reader_cls, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        page1 = MagicMock()
        page1.extract_text.return_value = "Real text"
        page2 = MagicMock()
        page2.extract_text.return_value = None
        mock_reader_cls.return_value.pages = [page1, page2]

        text = read_pdf(pdf_path)

        assert text == "Real text"

    @patch("assistant.tools.documents.PdfReader")
    def test_empty_pdf_returns_empty_string(self, mock_reader_cls, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        mock_reader_cls.return_value.pages = []

        assert read_pdf(pdf_path) == ""

    @patch("assistant.tools.documents.PdfReader")
    def test_corrupt_pdf_raises_document_error(self, mock_reader_cls, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"not a pdf")
        mock_reader_cls.side_effect = Exception("EOF marker not found")

        with pytest.raises(DocumentError) as exc_info:
            read_pdf(pdf_path)

        assert str(pdf_path) in str(exc_info.value)
        assert "EOF marker not found" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests for read_docx
# ---------------------------------------------------------------------------


class TestReadDocx:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_docx(tmp_path / "missing.docx")

    def test_missing_file_error_mentions_path(self, tmp_path):
        missing = tmp_path / "missing.docx"

        with pytest.raises(FileNotFoundError) as exc_info:
            read_docx(missing)

        assert str(missing) in str(exc_info.value)

    def test_extracts_paragraph_text(self, tmp_path):
        docx_path = tmp_path / "doc.docx"
        create_docx(docx_path, ["First paragraph", "Second paragraph"])

        text = read_docx(docx_path)

        assert text == "First paragraph\nSecond paragraph"

    def test_corrupt_docx_raises_document_error(self, tmp_path):
        docx_path = tmp_path / "doc.docx"
        docx_path.write_bytes(b"not a real docx file")

        with pytest.raises(DocumentError) as exc_info:
            read_docx(docx_path)

        assert str(docx_path) in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests for create_docx
# ---------------------------------------------------------------------------


class TestCreateDocx:
    def test_empty_string_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            create_docx(tmp_path / "doc.docx", "")

    def test_whitespace_only_string_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            create_docx(tmp_path / "doc.docx", "   ")

    def test_empty_list_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            create_docx(tmp_path / "doc.docx", [])

    def test_list_of_blank_strings_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            create_docx(tmp_path / "doc.docx", ["", "   "])

    def test_returns_path_and_writes_file(self, tmp_path):
        docx_path = tmp_path / "doc.docx"

        result = create_docx(docx_path, "Hello")

        assert result == docx_path
        assert docx_path.is_file()

    def test_creates_parent_directories(self, tmp_path):
        docx_path = tmp_path / "nested" / "dir" / "doc.docx"

        create_docx(docx_path, "Hello")

        assert docx_path.is_file()

    def test_string_content_split_into_paragraphs_by_newline(self, tmp_path):
        docx_path = tmp_path / "doc.docx"
        create_docx(docx_path, "Line one\nLine two\nLine three")

        text = read_docx(docx_path)

        assert text == "Line one\nLine two\nLine three"

    def test_list_content_preserved_in_order(self, tmp_path):
        docx_path = tmp_path / "doc.docx"
        create_docx(docx_path, ["Alpha", "Beta", "Gamma"])

        text = read_docx(docx_path)

        assert text == "Alpha\nBeta\nGamma"

    @patch("assistant.tools.documents.Document")
    def test_write_failure_raises_document_error(self, mock_document_cls, tmp_path):
        mock_doc = MagicMock()
        mock_doc.save.side_effect = OSError("disk full")
        mock_document_cls.return_value = mock_doc

        with pytest.raises(DocumentError) as exc_info:
            create_docx(tmp_path / "doc.docx", "Hello")

        assert "disk full" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
