from pathlib import Path

from app.ai.self_update import FeedItem, SourceDefinition, collect_and_update, parse_feed


def source() -> SourceDefinition:
    return SourceDefinition(
        name="Example",
        url="https://example.com/feed.xml",
        allowed_hosts=("example.com",),
        max_items=10,
    )


def item(title: str) -> FeedItem:
    return FeedItem(
        source="Example",
        title=title,
        url=f"https://example.com/{title.casefold().replace(' ', '-')}",
        summary=f"Summary for {title}.",
        published="2026-09-30T00:00:00Z",
        fingerprint=title,
    )


def test_parse_rss_filters_external_links_and_missing_summaries() -> None:
    content = b"""<?xml version="1.0"?>
    <rss version="2.0"><channel>
      <item><title>Trusted</title><link>https://example.com/trusted</link><description>Hello &lt;b&gt;world&lt;/b&gt;.</description></item>
      <item><title>External</title><link>https://other.example/item</link><description>No.</description></item>
      <item><title>No Summary</title><link>https://example.com/no-summary</link></item>
    </channel></rss>"""
    parsed = parse_feed(content, source())
    assert len(parsed) == 1
    assert parsed[0].title == "Trusted"
    assert parsed[0].summary == "Hello world."


def test_parse_atom_accepts_link_href() -> None:
    content = b"""<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Atom item</title>
        <link href="https://example.com/atom" />
        <summary>Atom summary</summary>
      </entry>
    </feed>"""
    parsed = parse_feed(content, source())
    assert parsed[0].url == "https://example.com/atom"


def test_collect_update_deduplicates_and_writes_state_and_reference(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.json"
    state_path = tmp_path / "state.json"
    knowledge_path = tmp_path / "auto_web.txt"
    sources_path.write_text(
        '[{"name":"Example","url":"https://example.com/feed.xml","allowed_hosts":["example.com"]}]',
        encoding="utf-8",
    )
    calls = [item("First"), item("Second")]

    report = collect_and_update(
        sources_path=sources_path,
        state_path=state_path,
        knowledge_path=knowledge_path,
        fetcher=lambda _source: calls,
    )
    assert report.changed is True
    assert report.new_items == 2
    assert state_path.exists()
    assert knowledge_path.exists()

    second = collect_and_update(
        sources_path=sources_path,
        state_path=state_path,
        knowledge_path=knowledge_path,
        fetcher=lambda _source: calls,
    )
    assert second.changed is False
    assert second.new_items == 0
    assert "First" in knowledge_path.read_text(encoding="utf-8")


def test_collect_update_fails_closed_without_wiping_existing_data(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.json"
    state_path = tmp_path / "state.json"
    knowledge_path = tmp_path / "auto_web.txt"
    sources_path.write_text(
        '[{"name":"Example","url":"https://example.com/feed.xml","allowed_hosts":["example.com"]}]',
        encoding="utf-8",
    )
    initial = [item("Keep me")]

    collect_and_update(
        sources_path=sources_path,
        state_path=state_path,
        knowledge_path=knowledge_path,
        fetcher=lambda _source: initial,
    )
    before_state = state_path.read_text(encoding="utf-8")
    before_knowledge = knowledge_path.read_text(encoding="utf-8")

    def broken(_source):
        raise RuntimeError("network down")

    report = collect_and_update(
        sources_path=sources_path,
        state_path=state_path,
        knowledge_path=knowledge_path,
        fetcher=broken,
    )
    assert report.changed is False
    assert report.source_errors
    assert state_path.read_text(encoding="utf-8") == before_state
    assert knowledge_path.read_text(encoding="utf-8") == before_knowledge
