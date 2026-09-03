"""Loading the knowledge base off disk.

An article is markdown with a small front matter block. Each ``##`` heading
becomes a separately retrievable :class:`Passage`, because a customer asking
"how do I get a return label" wants one section, not a four-section article.
"""

from __future__ import annotations

from pathlib import Path

from ..models import Intent, Passage

ARTICLE_DIR = Path(__file__).parent / "articles"


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw
    _, _, rest = raw.partition("---\n")
    block, sep, body = rest.partition("---\n")
    if not sep:
        return {}, raw
    meta: dict[str, str] = {}
    for line in block.splitlines():
        key, _, value = line.partition(":")
        if value:
            meta[key.strip()] = value.strip()
    return meta, body


def _parse_intents(raw: str) -> tuple[Intent, ...]:
    intents = []
    for name in raw.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            intents.append(Intent(name))
        except ValueError:
            continue  # an article tagged with an intent we no longer have
    return tuple(intents)


def parse_article(raw: str) -> list[Passage]:
    meta, body = _parse_front_matter(raw)
    article_id = meta.get("id", "")
    title = meta.get("title", article_id)
    url = meta.get("url", "")
    intents = _parse_intents(meta.get("intents", ""))
    human_only = {
        name.strip().lower()
        for name in meta.get("handoff_sections", "").split(";")
        if name.strip()
    }

    passages: list[Passage] = []
    section = ""
    lines: list[str] = []

    def flush() -> None:
        text = "\n".join(lines).strip()
        if section and text:
            passages.append(
                Passage(
                    article_id=article_id,
                    article_title=title,
                    section=section,
                    text=text,
                    url=url,
                    intents=intents,
                    requires_human=section.strip().lower() in human_only,
                )
            )

    for line in body.splitlines():
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
            lines = []
        else:
            lines.append(line)
    flush()
    return passages


def load_passages(directory: Path | None = None) -> list[Passage]:
    """Read every article in ``directory`` into passages, sorted for stability."""
    directory = directory or ARTICLE_DIR
    passages: list[Passage] = []
    for path in sorted(directory.glob("*.md")):
        passages.extend(parse_article(path.read_text(encoding="utf-8")))
    return passages
