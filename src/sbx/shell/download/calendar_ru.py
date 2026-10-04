"""Производственный календарь РФ (xmlcalendar.ru) — сырые данные для календарных признаков."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import certifi

URL = "https://xmlcalendar.ru/data/ru/{year}/calendar.json"


def fetch_year(year: int, timeout: int = 60) -> dict:
    request = urllib.request.Request(
        URL.format(year=year), headers={"User-Agent": "sbx-research/0.1"}
    )
    with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
        return json.loads(response.read().decode("utf-8"))


def save_calendar(path: Path, years: Sequence[int]) -> int:
    data = {str(year): fetch_year(year) for year in years}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(data)


def ssl_context():
    """CA-хранилище certifi: в изолированном окружении uv системные сертификаты недоступны."""
    import ssl

    return ssl.create_default_context(cafile=certifi.where())
