"""PubMed retrieval module.

Wraps the NCBI Entrez E-Utilities (``esearch`` + ``efetch``) for the V1
pipeline. The roadmap recommends Biopython, but for a coursework V1 a
thin HTTP + ElementTree layer is more readable, has no extra
dependencies, and is easier to mock in tests.

NCBI usage policy:
  * Always identify the application via the ``tool`` parameter.
  * Provide an email so NCBI can contact you on rate-limit issues.
  * With an API key, free tier is 10 requests/sec; without, 3 req/sec.
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

from config import Settings, load_settings
from src.http_utils import HTTPRetryError, request_with_retry

logger = logging.getLogger(__name__)

ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "ai-research-agent-v1"


@dataclass
class Article:
    """A single PubMed article record (abstract-level for V1)."""

    pmid: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    publication_types: List[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def top_study_type(self) -> str:
        """Best label for the UI (RCT / Meta-analysis / Review / Other)."""
        if not self.publication_types:
            return "Other"
        types_lower = " ".join(self.publication_types).lower()
        if "meta-analysis" in types_lower:
            return "Meta-analysis"
        if "systematic review" in types_lower:
            return "Systematic review"
        if "randomized controlled trial" in types_lower:
            return "RCT"
        if "review" in types_lower:
            return "Review"
        if "clinical trial" in types_lower:
            return "Clinical trial"
        if "cohort" in types_lower:
            return "Cohort"
        if "case-control" in types_lower or "case control" in types_lower:
            return "Case-control"
        return "Other"

    def authors_short(self, n: int = 3) -> str:
        """Return a short author string like "Smith J, Doe A et al.".

        ``n`` is clamped to at least 1 so callers cannot accidentally
        produce an "et al."-only string.
        """
        if not self.authors:
            return ""
        n = max(1, int(n))
        if len(self.authors) <= n:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:n]) + " et al."


class PubMedError(RuntimeError):
    """Raised when PubMed retrieval fails irrecoverably."""


class PubMedRetriever:
    """Minimal Entrez E-Utilities client for V1."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or load_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(self, query: str, max_results: Optional[int] = None) -> List[Article]:
        """Search PubMed and return up to ``max_results`` parsed articles."""
        limit = max_results or self.settings.max_articles
        pmids = self._esearch(query, limit)
        if not pmids:
            return []
        # Brief courtesy pause between esearch and efetch.
        time.sleep(0.1)
        return self._efetch(pmids)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _base_params(self) -> dict:
        params = {"tool": TOOL_NAME}
        if self.settings.ncbi_email:
            params["email"] = self.settings.ncbi_email
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        return params

    def _esearch(self, query: str, retmax: int) -> List[str]:
        params = {
            **self._base_params(),
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(retmax),
            "sort": "relevance",
        }
        try:
            resp = request_with_retry(
                "GET",
                f"{ENTREZ_BASE}/esearch.fcgi",
                params=params,
                timeout=self.settings.request_timeout_seconds,
            )
        except HTTPRetryError as exc:
            raise PubMedError(f"PubMed search failed: {exc}") from exc
        if resp.status_code >= 400:
            raise PubMedError(
                f"PubMed esearch HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            data = resp.json()
            return list(data["esearchresult"]["idlist"])
        except (ValueError, KeyError) as exc:
            raise PubMedError(f"Malformed esearch response: {exc}") from exc

    def _efetch(self, pmids: List[str]) -> List[Article]:
        params = {
            **self._base_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        try:
            resp = request_with_retry(
                "GET",
                f"{ENTREZ_BASE}/efetch.fcgi",
                params=params,
                timeout=self.settings.request_timeout_seconds,
            )
        except HTTPRetryError as exc:
            raise PubMedError(f"PubMed fetch failed: {exc}") from exc
        if resp.status_code >= 400:
            raise PubMedError(
                f"PubMed efetch HTTP {resp.status_code}: {resp.text[:300]}"
            )
        return parse_pubmed_xml(resp.text)


# ----------------------------------------------------------------------
# Parsing helpers (exposed for unit tests)
# ----------------------------------------------------------------------
def _text(el: Optional[ET.Element]) -> str:
    return (el.text or "").strip() if el is not None and el.text else ""


def _collect_abstract(article_el: ET.Element) -> str:
    """PubMed abstracts can be split into multiple labelled sections."""
    parts: List[str] = []
    for ab in article_el.iter("AbstractText"):
        label = ab.attrib.get("Label")
        text = "".join(ab.itertext()).strip()
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts).strip()


def _collect_authors(article_el: ET.Element) -> List[str]:
    authors: List[str] = []
    for author in article_el.iter("Author"):
        last = _text(author.find("LastName"))
        initials = _text(author.find("Initials"))
        collective = _text(author.find("CollectiveName"))
        if collective:
            authors.append(collective)
        elif last:
            authors.append(f"{last} {initials}".strip())
    return authors


def _collect_publication_types(article_el: ET.Element) -> List[str]:
    return [
        "".join(pt.itertext()).strip()
        for pt in article_el.iter("PublicationType")
        if "".join(pt.itertext()).strip()
    ]


def _extract_year(article_el: ET.Element) -> str:
    """Return the publication year as a 4-char string.

    Tries <PubDate> first (most reliable), then <MedlineDate> like
    "2023 Mar-Apr", then <ArticleDate> as a last resort. Returns ""
    if no usable year is present.
    """
    for date_el in article_el.iter("PubDate"):
        year = _text(date_el.find("Year"))
        if year:
            return year
        medline = _text(date_el.find("MedlineDate"))
        if medline:
            for token in medline.split():
                if token[:4].isdigit():
                    return token[:4]
    # Some records carry only <ArticleDate><Year>...</Year></ArticleDate>
    for date_el in article_el.iter("ArticleDate"):
        year = _text(date_el.find("Year"))
        if year:
            return year
    return ""


def parse_pubmed_xml(xml_text: str) -> List[Article]:
    """Parse an ``efetch`` PubMed XML response into ``Article`` objects.

    Exposed as a module-level function so unit tests can feed canned XML.
    """
    if not xml_text or not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise PubMedError(f"Could not parse PubMed XML: {exc}") from exc

    articles: List[Article] = []
    for pa in root.iter("PubmedArticle"):
        pmid = _text(pa.find(".//PMID"))
        if not pmid:
            continue
        title_el = pa.find(".//ArticleTitle")
        title = (
            "".join(title_el.itertext()).strip() if title_el is not None else ""
        )
        abstract = _collect_abstract(pa)
        authors = _collect_authors(pa)
        journal = _text(pa.find(".//Journal/Title"))
        year = _extract_year(pa)
        pub_types = _collect_publication_types(pa)
        articles.append(
            Article(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                year=year,
                publication_types=pub_types,
            )
        )
    return articles
