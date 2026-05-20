"""Tests for the PubMed retriever, using canned XML so we never hit the network."""
from __future__ import annotations

from src.pubmed_retriever import parse_pubmed_xml


SAMPLE_XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal>
          <Title>Journal of Imaginary Medicine</Title>
        </Journal>
        <ArticleTitle>Effect of widget on outcome in adults</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Widgets are common.</AbstractText>
          <AbstractText Label="METHODS">We ran a randomized trial.</AbstractText>
          <AbstractText Label="RESULTS">Widgets improved outcomes.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><Initials>J</Initials></Author>
          <Author><LastName>Doe</LastName><Initials>A</Initials></Author>
        </AuthorList>
        <PublicationTypeList>
          <PublicationType>Randomized Controlled Trial</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <History>
        <PubMedPubDate PubStatus="pubmed">
          <Year>2023</Year>
        </PubMedPubDate>
      </History>
    </PubmedData>
    <MedlineJournalInfo></MedlineJournalInfo>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>22222222</PMID>
      <Article>
        <Journal><Title>Lancet</Title></Journal>
        <ArticleTitle>Another study</ArticleTitle>
        <Abstract>
          <AbstractText>Single abstract paragraph here.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def test_parse_pubmed_xml_returns_two_articles():
    articles = parse_pubmed_xml(SAMPLE_XML)
    assert len(articles) == 2


def test_parse_pubmed_xml_labelled_sections():
    articles = parse_pubmed_xml(SAMPLE_XML)
    first = articles[0]
    assert first.pmid == "12345678"
    assert first.title.startswith("Effect of widget")
    assert "BACKGROUND" in first.abstract
    assert "METHODS" in first.abstract
    assert "RESULTS" in first.abstract
    assert first.authors == ["Smith J", "Doe A"]
    assert "Randomized Controlled Trial" in first.publication_types


def test_parse_pubmed_xml_single_paragraph_abstract():
    articles = parse_pubmed_xml(SAMPLE_XML)
    second = articles[1]
    assert second.pmid == "22222222"
    assert "Single abstract" in second.abstract


def test_parse_pubmed_xml_empty_input():
    assert parse_pubmed_xml("") == []


def test_article_url_and_authors_short():
    articles = parse_pubmed_xml(SAMPLE_XML)
    first = articles[0]
    assert first.url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    # Two authors -> all listed
    assert first.authors_short(n=3) == "Smith J, Doe A"


ARTICLE_DATE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>987</PMID>
      <Article>
        <Journal><Title>J</Title></Journal>
        <ArticleTitle>Year-fallback test</ArticleTitle>
        <Abstract><AbstractText>An abstract.</AbstractText></Abstract>
        <ArticleDate><Year>2024</Year><Month>03</Month><Day>15</Day></ArticleDate>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def test_year_extracted_from_article_date_when_pub_date_missing():
    arts = parse_pubmed_xml(ARTICLE_DATE_XML)
    assert arts and arts[0].year == "2024"


def test_authors_short_clamps_non_positive_n():
    from src.pubmed_retriever import Article

    a = Article(pmid="1", title="", abstract="", authors=["A", "B", "C"])
    # n<=0 should be treated as n=1 (returns first author + et al.)
    assert a.authors_short(0) == "A et al."
    assert a.authors_short(-7) == "A et al."
