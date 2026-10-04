"""Рендер лендинга на минимальных данных отчёта."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sbx.shell.render import landing


def _data() -> dict[str, Any]:
    return {
        "meta": {
            "title": "Прогноз расходов МО",
            "subtitle": "Панель: 2 ряда.",
            "cards": [{"value": "500,0", "label": "MAE на горизонте 1 месяц, руб."}],
            "winners": [
                {"horizon": 1, "model": "Ensemble", "mae": 500.0, "reference_mae": 600.0},
                {"horizon": 12, "model": "Ensemble", "mae": 700.0, "reference_mae": 900.0},
            ],
            "horizon_coverage": {
                1: {"models": 3, "observations": 10},
                12: {"models": 2, "observations": 30},
            },
            "limitations": "История каждого ряда — 24 месяца.",
            "footer": "Сформировано автоматически.",
        },
        "leaderboard": [],
        "leaderboard_view": [
            {"Модель": "Ensemble", "MAE, руб.": 600.0},
            {"Модель": "Prophet", "MAE, руб.": 750.0},
        ],
        "horizon_leaderboard": [
            {"Модель": "Ensemble", "mae h=1": 500.0, "mae h=12": 700.0},
            {"Модель": "Prophet", "mae h=1": 600.0, "mae h=12": 900.0},
            # Модель, неопределимая на длинном горизонте.
            {"Модель": "LightGBM", "mae h=1": 550.0, "mae h=12": float("nan")},
        ],
    }


def _page(path: Path) -> str:
    """Собранная страница с обычными пробелами вместо неразрывных между разрядами чисел."""
    return path.read_text(encoding="utf-8").replace("\u00a0", " ")


def _render(tmp_path: Path) -> str:
    return _page(landing.render(_data(), tmp_path / "index.html"))


def _visible(html: str) -> str:
    """Разметка страницы без таблицы стилей и сценария: в шрифтах встречается что угодно."""
    return re.sub(r"<(style|script)\b.*?</\1>", "", html, flags=re.S)


def test_landing_renders_the_horizon_table_and_the_data_source(tmp_path: Path) -> None:
    html = _render(tmp_path)
    assert "Лучшая модель зависит от горизонта прогноза" in html
    # Лидер каждого горизонта выделен, эталон подсвечен, пропуск назван словами.
    assert html.count('<td class="win">') == 2
    assert (
        '<tr class="reference"><th scope="row">Prophet <span class="tag-ref">эталон</span></th>'
        in html
    )
    assert html.count('<td class="na">не строится¹</td>') == 1
    assert "¹</b> Не строится: глобальному LightGBM при h=12" in html, "сноска объясняет пропуск"
    assert "nan" not in _visible(html).lower()
    assert "<td>3 / 10</td><td>2 / 30</td>" in html, "под таблицей — число моделей и наблюдений"
    # Подпись источника данных обязательна по условиям конкурса.
    assert "Данные СберИндекса" in html


def test_horizon_table_groups_models_by_family(tmp_path: Path) -> None:
    html = _render(tmp_path)
    table = html[html.index('<table class="board">') :].split("</table>")[0]
    groups = re.findall(r'<tr class="fam"><th colspan="3" scope="colgroup">([^<]+)</th>', table)
    assert groups == ["Машинное обучение и нейросети", "Ансамбли", "Эталон и наивные прогнозы"]
    leaders = table[table.index('<tr class="leaders">') :].split("</tr>")[0]
    assert leaders.count("<span>Ансамбль</span>") == 2 and "<b>500,0</b>" in leaders
    assert "окон проверки" not in table, "счёта окон нет — нет и строки о нём"


def test_minimal_page_has_no_blocks_it_has_no_data_for(tmp_path: Path) -> None:
    html = _render(tmp_path)
    anchors = set(re.findall(r'<section id="([^"]+)"', html))
    assert anchors == {"pipeline", "horizons", "metrics", "limits", "reproduce"}
    assert "Архитектура решения" in html and "Functional Core / Imperative Shell" in html
    assert '<article class="limit">' in html and "История каждого ряда — 24 месяца." in html
    assert 'href="methodology.pdf"' not in html and "Репозиторий" not in html


def test_header_links_to_the_report_and_the_repository_when_they_are_given(tmp_path: Path) -> None:
    data = _data()
    data["meta"] |= {
        "pdf": "methodology.pdf",
        "repository": "https://example.org/repo",
        "contest": "Онлайн-конкурс СберИндекса 2026 · направление «Прогнозирование»",
        "panel_line": "Панель: 2 ряда.",
    }
    html = _page(landing.render(data, tmp_path / "links.html"))
    header = html[html.index('<header class="top">') :].split("</header>")[0]
    assert '<a href="methodology.pdf">Методологический отчёт, PDF</a>' in header
    assert '<a href="https://example.org/repo">Репозиторий</a>' in header
    assert '<a href="#reproduce">Как воспроизвести</a>' in header
    hero = html[html.index('<div class="hero"') :].split("</div>")[0]
    assert (
        '<p class="eyebrow">Онлайн-конкурс СберИндекса 2026 · направление «Прогнозирование»</p>'
        in hero
    )
    assert '<p class="hero-lead">Панель: 2 ряда.</p>' in hero, (
        "вводная — без повтора названия конкурса"
    )
    footer = html[html.index("<footer>") :]
    assert 'href="methodology.pdf"' in footer and '<a href="#top">Наверх</a>' in footer


def test_landing_declares_an_icon_so_browsers_do_not_request_favicon(tmp_path: Path) -> None:
    """Без объявленной иконки браузер запрашивает /favicon.ico и пишет 404 в консоль."""
    head = _render(tmp_path).split("</head>")[0]
    assert '<link rel="icon"' in head


def test_landing_points_to_the_full_run_command(tmp_path: Path) -> None:
    """Команды воспроизведения — обязательное содержание страницы; неполный список шагов
    даёт другой отчёт, поэтому страница называет одну команду полного прогона."""
    html = _render(tmp_path)
    assert "<code>make all</code>" in html


def test_landing_says_so_when_it_was_built_without_the_news_archive(tmp_path: Path) -> None:
    data = _data()
    data["ablations"] = [{"name": "A", "mae": 700.0, "delta_mae": 0.0}]
    html = _page(landing.render(data, tmp_path / "with.html"))
    assert "Абляции: что реально помогает" in html
    assert "архива новостей GDELT не было" not in html

    data["meta"]["news_note"] = "В этом прогоне архива новостей GDELT не было."
    html = _page(landing.render(data, tmp_path / "without.html"))
    assert "В этом прогоне архива новостей GDELT не было." in html


def test_ablation_text_states_what_was_measured_not_a_fixed_conclusion(tmp_path: Path) -> None:
    """Вывод о вкладе внешних данных собирается из знака и значимости разницы MAE."""
    data = _data()
    data["ablations"] = [{"name": "A", "mae": 700.0, "delta_mae": 0.0}]
    data["meta"]["contributions"] = {
        "diebold_mariano": {
            "B_vs_A": {
                "mean_diff": -60.8,
                "p_value": 2.0e-9,
                "folds": 10,
                "folds_better": 9,
                "folds_worse": 1,
                "sign_p": 0.021,
            },
            "C_vs_B": {
                "mean_diff": 6.7,
                "p_value": 0.31,
                "folds": 10,
                "folds_better": 4,
                "folds_worse": 6,
                "sign_p": 0.754,
            },
        }
    }
    html = _page(landing.render(data, tmp_path / "helps.html"))
    assert "Календарь и национальные ряды: MAE ниже на 60,8 руб." in html
    assert "устойчивого изменения нет" in html
    assert "Внешние признаки ухудшают прогноз" not in html

    data["meta"]["contributions"]["diebold_mariano"]["B_vs_A"] = {
        "mean_diff": 111.9,
        "p_value": 1e-12,
    }
    html = _page(landing.render(data, tmp_path / "hurts.html"))
    assert "Календарь и национальные ряды: MAE выше на 111,9 руб." in html
