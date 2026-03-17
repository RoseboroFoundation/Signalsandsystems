"""Tests for clean/party_platforms.py — platform text extraction."""

from unittest.mock import patch, MagicMock

import pytest

from clean.party_platforms import PartyPlatformDownloader


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def downloader(tmp_path):
    return PartyPlatformDownloader(output_dir=str(tmp_path / "platforms"))


@pytest.fixture
def app_html():
    """Minimal American Presidency Project HTML page with platform text."""
    return """
    <html>
    <body>
    <article>
        <div class="field-docs-content">
            <h1>2024 Republican Party Platform</h1>
            <p>We believe in freedom and liberty for all Americans.</p>
            <p>The economy must grow through lower taxes and less regulation.</p>
            <p>National security is our top priority.</p>
            <li>Secure the border</li>
            <li>Strengthen the military</li>
        </div>
    </article>
    </body>
    </html>
    """


@pytest.fixture
def app_html_alt_class():
    """APP HTML using the alternate CSS class name."""
    return """
    <html>
    <body>
    <div class="field--name-field-docs-content">
        <p>This is the platform text from the alternate class.</p>
        <p>Second paragraph of content.</p>
    </div>
    </body>
    </html>
    """


@pytest.fixture
def app_html_article_fallback():
    """APP HTML with no field-docs-content, falls back to <article>."""
    return """
    <html>
    <body>
    <article>
        <p>Fallback platform text inside article tag.</p>
    </article>
    </body>
    </html>
    """


@pytest.fixture
def app_html_no_content():
    """APP HTML with no recognizable content container."""
    return """
    <html>
    <body>
    <div class="other-class">
        <p>This should not be found.</p>
    </div>
    </body>
    </html>
    """


# ── _extract_text_from_app ──────────────────────────────────────────────

class TestExtractTextFromApp:
    """Test HTML parsing of American Presidency Project pages."""

    def test_extracts_paragraphs_and_headings(self, downloader, app_html):
        text = downloader._extract_text_from_app(app_html)
        assert text is not None
        assert "freedom and liberty" in text
        assert "economy must grow" in text
        assert "National security" in text

    def test_extracts_list_items(self, downloader, app_html):
        text = downloader._extract_text_from_app(app_html)
        assert "Secure the border" in text
        assert "Strengthen the military" in text

    def test_extracts_heading(self, downloader, app_html):
        text = downloader._extract_text_from_app(app_html)
        assert "2024 Republican Party Platform" in text

    def test_alt_class_name(self, downloader, app_html_alt_class):
        text = downloader._extract_text_from_app(app_html_alt_class)
        assert text is not None
        assert "alternate class" in text

    def test_article_fallback(self, downloader, app_html_article_fallback):
        text = downloader._extract_text_from_app(app_html_article_fallback)
        assert text is not None
        assert "Fallback platform text" in text

    def test_returns_none_when_no_content(self, downloader, app_html_no_content):
        text = downloader._extract_text_from_app(app_html_no_content)
        assert text is None

    def test_paragraphs_separated_by_newlines(self, downloader, app_html):
        text = downloader._extract_text_from_app(app_html)
        # Each paragraph should be on its own line
        lines = [l for l in text.split('\n\n') if l.strip()]
        assert len(lines) >= 3


# ── _extract_text_from_pdf ──────────────────────────────────────────────

class TestExtractTextFromPdf:
    """Test PDF text extraction with mocked pypdf."""

    def test_extracts_text_from_pages(self, downloader):
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page one content."
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page two content."

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]

        with patch('clean.party_platforms._try_import_pypdf') as mock_import:
            MockPdfReader = MagicMock(return_value=mock_reader)
            mock_import.return_value = MockPdfReader
            text = downloader._extract_text_from_pdf(b'%PDF-fake')

        assert text is not None
        assert "Page one content." in text
        assert "Page two content." in text
        assert "\n\n" in text  # pages joined with double newline

    def test_returns_none_when_no_text_extracted(self, downloader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch('clean.party_platforms._try_import_pypdf') as mock_import:
            MockPdfReader = MagicMock(return_value=mock_reader)
            mock_import.return_value = MockPdfReader
            text = downloader._extract_text_from_pdf(b'%PDF-fake')

        assert text is None

    def test_returns_none_when_pypdf_not_installed(self, downloader):
        with patch('clean.party_platforms._try_import_pypdf', return_value=None):
            text = downloader._extract_text_from_pdf(b'%PDF-fake')

        assert text is None


# ── download_platform (integration with mocked HTTP) ────────────────────

class TestDownloadPlatform:
    """Test the download_platform orchestrator with mocked requests."""

    def test_successful_html_download(self, downloader, app_html):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = app_html

        with patch.object(downloader, '_request', return_value=mock_resp):
            result = downloader.download_platform(2000, 'Republican')

        assert result is not None
        assert result['year'] == 2000
        assert result['party'] == 'Republican'
        assert result['word_count'] > 0
        assert "freedom and liberty" in result['text']

    def test_returns_none_for_unknown_year(self, downloader):
        result = downloader.download_platform(1999, 'Republican')
        assert result is None

    def test_returns_none_when_all_sources_fail(self, downloader):
        with patch.object(downloader, '_request', return_value=None):
            result = downloader.download_platform(2024, 'Republican')

        assert result is None

    def test_2020_republican_note(self, downloader, app_html):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b'%PDF-fake'

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "We the people..."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch.object(downloader, '_request', return_value=mock_resp), \
             patch('clean.party_platforms._try_import_pypdf') as mock_import:
            MockPdfReader = MagicMock(return_value=mock_reader)
            mock_import.return_value = MockPdfReader
            result = downloader.download_platform(2020, 'Republican')

        assert result is not None
        assert 'Re-adopted 2016' in result['note']

    def test_pdf_fallback_to_html(self, downloader, app_html):
        """When PDF extraction fails, should try HTML source."""
        pdf_resp = MagicMock()
        pdf_resp.status_code = 200
        pdf_resp.content = b'%PDF-bad'

        html_resp = MagicMock()
        html_resp.status_code = 200
        html_resp.text = app_html

        call_count = 0

        def side_effect(url):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pdf_resp  # first call = PDF source
            return html_resp  # second call = APP fallback

        with patch.object(downloader, '_request', side_effect=side_effect), \
             patch('clean.party_platforms._try_import_pypdf', return_value=None):
            result = downloader.download_platform(2016, 'Republican')

        assert result is not None
        assert result['source'] == 'APP'
