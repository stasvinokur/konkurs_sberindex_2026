"""Справочник ОКТМО муниципального уровня из Wikidata (свойство P764)."""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
from pathlib import Path

import certifi
import pandas as pd

SPARQL_URL = "https://query.wikidata.org/sparql"
QUERY = """
SELECT ?item ?oktmo ?label WHERE {
  ?item wdt:P764 ?oktmo .
  FILTER(STRLEN(?oktmo) = 8 && STRENDS(?oktmo, "000"))
  ?item rdfs:label ?label . FILTER(LANG(?label) = "ru")
}
"""


def fetch_reference(timeout: int = 180) -> pd.DataFrame:
    url = SPARQL_URL + "?" + urllib.parse.urlencode({"query": QUERY})
    request = urllib.request.Request(
        url, headers={"Accept": "text/csv", "User-Agent": "sbx-research/0.1"}
    )
    with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
        df = pd.read_csv(io.BytesIO(response.read()), dtype=str)
    df["item"] = df["item"].str.replace("http://www.wikidata.org/entity/", "", regex=False)
    return df.sort_values(["oktmo", "label"]).drop_duplicates().reset_index(drop=True)


def save_reference(path: Path) -> int:
    df = fetch_reference()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)


def ssl_context():
    """CA-хранилище certifi: в изолированном окружении uv системные сертификаты недоступны."""
    import ssl

    return ssl.create_default_context(cafile=certifi.where())
