import httpx

from pipelines.indexing import CorpusIndex
from pipelines.ingestion.dedup import deduplicate, normalize_title
from pipelines.ingestion.openalex import (
    OpenAlexClient, normalize_doi, reconstruct_abstract, work_to_paper)
from pipelines.ingestion.schema import Paper


def make_work(i="W1", title="A Study of Deep Learning Methods", doi="https://doi.org/10.1/ABC",
              year=2022, abstract=None, oa=True):
    return {
        "id": f"https://openalex.org/{i}", "doi": doi, "title": title, "publication_year": year,
        "abstract_inverted_index": abstract,
        "authorships": [{"author": {"display_name": "Ada L."}}, {"author": {"display_name": None}}],
        "primary_location": {"landing_page_url": "https://example.org/p", "license": "cc-by"},
        "open_access": {"is_oa": oa, "oa_url": "https://example.org/p.pdf" if oa else None},
    }


def P(i, title="A Study of Deep Learning Methods", doi=None, year=2022, abstract=None, domains=()):
    return Paper(paper_id=f"oa_{i}", source_id=i, title=title, doi=doi,
                 publication_year=year, abstract=abstract, domains=list(domains))


def test_reconstruct_abstract_orders_words_by_position():
    inv = {"world": [1], "Hello": [0], "again": [2, 3]}
    assert reconstruct_abstract(inv) == "Hello world again again"
    assert reconstruct_abstract(None) is None


def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1/ABC") == "10.1/abc"
    assert normalize_doi(None) is None


def test_work_to_paper_mapping_and_missing_title():
    p = work_to_paper(make_work(abstract={"Hi": [0]}), "computer vision")
    assert p.paper_id == "oa_W1" and p.doi == "10.1/abc" and p.authors == ["Ada L."]
    assert p.abstract == "Hi" and p.full_text_status == "open_access_link"
    assert p.domains == ["computer vision"]
    assert work_to_paper(make_work(title="  ")) is None
    assert work_to_paper(make_work(oa=False)).full_text_status == "metadata_only"


def test_dedup_same_doi_merges_and_unions_domains():
    a = P("1", doi="10.1/x", domains=["machine learning"])
    b = P("2", doi="10.1/x", abstract="has abstract", domains=["computer vision"])
    unique, log, possible = deduplicate([a, b])
    assert len(unique) == 1 and not possible
    assert unique[0].abstract == "has abstract"
    assert unique[0].domains == ["computer vision", "machine learning"]


def test_dedup_same_title_different_doi_is_flagged_not_merged():
    unique, log, possible = deduplicate([P("1", doi="10.1/a"), P("2", doi="10.2/b")])
    assert len(unique) == 2 and not log
    assert possible[0]["reason"] == "same_title_different_doi"


def test_dedup_same_title_missing_doi_merges():
    unique, log, _ = deduplicate([P("1", doi="10.1/a"), P("2", doi=None)])
    assert len(unique) == 1 and log[0]["reason"] == "same_title"


def test_short_titles_never_match():
    unique, _, _ = deduplicate([P("1", title="Introduction"), P("2", title="Introduction")])
    assert len(unique) == 2


def test_normalize_title_ignores_case_accents_punctuation():
    assert normalize_title("Déjà-Vu: A Study!") == normalize_title("deja vu a study")


def test_client_follows_cursor_and_respects_max_results():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        cursor = request.url.params["cursor"]
        if cursor == "*":
            return httpx.Response(200, json={"results": [make_work("W1"), make_work("W2")],
                                             "meta": {"next_cursor": "c2"}})
        return httpx.Response(200, json={"results": [make_work("W3"), make_work("W4")],
                                         "meta": {"next_cursor": None}})

    c = OpenAlexClient(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    got = list(c.search_works("q", 2018, ["article", "preprint"], max_results=3))
    assert [w["id"][-2:] for w in got] == ["W1", "W2", "W3"]
    assert calls[0]["api_key"] == "k" and "type:article|preprint" in calls[0]["filter"]


def test_client_retries_on_429_then_succeeds():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"results": [], "meta": {}})

    waits = []
    c = OpenAlexClient(client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=waits.append)
    assert list(c.search_works("q", 2018, ["article"], 5)) == []
    assert waits == [1, 2]


def test_corpus_index_ranks_title_matches_and_applies_domain_filter():
    docs = [
        {
            "paper_id": "p1",
            "title": "Deep Learning for Computer Vision",
            "abstract": "Using neural networks to recognise visual patterns.",
            "domains": ["machine learning", "computer vision"],
        },
        {
            "paper_id": "p2",
            "title": "Transformer Language Models",
            "abstract": "Large language models trained on textual corpora.",
            "domains": ["machine learning"],
        },
        {
            "paper_id": "p3",
            "title": "Computer Vision Object Detection in Images",
            "abstract": "A benchmark for detecting objects with CNNs.",
            "domains": ["computer vision"],
        },
    ]

    index = CorpusIndex(docs)
    results = index.search("deep learning", limit=5)
    assert results[0]["paper_id"] == "p1"
    assert results[0]["score"] > 0

    filtered = index.search("vision", limit=5, domain="computer vision")
    assert {item["paper_id"] for item in filtered} == {"p1", "p3"}
