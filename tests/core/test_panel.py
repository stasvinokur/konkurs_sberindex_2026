import numpy as np
import pandas as pd
import pytest

from sbx.core.panel import (
    PanelContractError,
    additivity,
    build_official_panel,
    build_panel,
    static_features,
    validate_panel,
)
from sbx.core.seasonality import (
    national_seasonal_index,
    ratio_to_moving_average,
    seasonal_index_from_ratios,
)


def _raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    periods = [f"2023-{m:02d}-01" for m in range(1, 13)] + [
        f"2024-{m:02d}-01" for m in range(1, 13)
    ]
    rows = []
    for mo, level, n in [("A", 1000.0, 24), ("B", 2000.0, 20)]:
        for cat, share in [("Все категории", 1.0), ("Продовольствие", 0.5), ("Здоровье", 0.1)]:
            for p in periods[-n:]:
                rows.append({"period": p, "mo": mo, "category_15": cat, "value": level * share})
    raw = pd.DataFrame(rows)
    blocks = raw.assign(
        block_id=(raw[["mo", "category_15"]] != raw[["mo", "category_15"]].shift())
        .any(axis=1)
        .cumsum()
    )
    bt = blocks.drop_duplicates("block_id")[["block_id", "mo"]].assign(
        territory_id=lambda d: "46-" + d["mo"].str.lower(),
        admin_break_at=lambda d: np.where(d["mo"] == "B", "2024-05-01", None),
    )
    territories = pd.DataFrame(
        {
            "territory_id": ["46-a", "46-b"],
            "raw_mo": ["городской округ A", "B муниципальный район"],
            "region_code": ["46", "46"],
            "oktmo": ["46701000", None],
        }
    )
    return raw, bt, territories


def test_build_panel_contract_and_flags() -> None:
    raw, bt, _ = _raw()
    panel = build_panel(raw, bt)
    assert len(panel) == len(raw)
    assert panel["unique_id"].nunique() == 6
    assert not panel.duplicated(["unique_id", "ds"]).any()
    b = panel[panel["territory_id"] == "46-b"]
    assert (~b["is_complete"]).all() and (b["n_obs"] == 20).all()
    assert (b["admin_break_at"] == pd.Timestamp("2024-05-01")).all()
    assert panel.loc[panel["territory_id"] == "46-a", "is_complete"].all()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.assign(y=p["y"].where(p.index != 0, -1.0)),
        lambda p: pd.concat([p, p.iloc[[0]]]),
        lambda p: p.assign(y=p["y"].where(p.index != 1, np.nan)),
        lambda p: p.assign(ds=p["ds"].astype(str)),
    ],
)
def test_validate_panel_rejects_bad_data(mutate) -> None:
    raw, bt, _ = _raw()
    panel = build_panel(raw, bt)
    with pytest.raises(PanelContractError):
        validate_panel(mutate(panel))


def test_static_features_and_additivity() -> None:
    raw, bt, territories = _raw()
    panel = build_panel(raw, bt)
    static = static_features(panel, territories, n_size_groups=2)
    assert len(static) == 6
    a = static[static["territory_id"] == "46-a"].iloc[0]
    assert a["region_name"] == "Московская область" and a["federal_district"] == "Центральный"
    assert a["mo_kind"] == "urban_okrug"
    assert np.isclose(a["size_level"], np.log(1000.0))
    assert set(static["size_group"]) == {1, 2}
    add = additivity(panel)
    assert np.isclose(add["ratio_median"], 0.6) and add["share_sum_exceeds_total"] == 0.0


def test_seasonal_index_recovers_known_pattern() -> None:
    ds = pd.date_range("2018-01-01", "2025-12-01", freq="MS")
    true = np.array([0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2])
    true = true / true.mean()
    trend = np.linspace(100, 160, len(ds))
    values = trend * true[ds.month - 1]
    idx = seasonal_index_from_ratios(ratio_to_moving_average(pd.Series(values, index=ds)))
    assert np.allclose(idx.to_numpy(), true, atol=0.01)
    assert np.isclose(idx.mean(), 1.0)

    national = pd.DataFrame({"period": ds.strftime("%Y-%m-%d"), "type": "Всего", "value": values})
    out = national_seasonal_index(national, end="2025-12-01")
    assert set(out["category"]) == {"Все категории"} and len(out) == 12


def _official() -> tuple[pd.DataFrame, pd.DataFrame]:
    months = [f"2023-{m:02d}" for m in range(1, 13)] + [f"2024-{m:02d}" for m in range(1, 13)]
    rows = []
    for official_id, level, n in [(1, 1000, 24), (2594, 2000, 20)]:
        for cat, share in [("Все категории", 1.0), ("Продовольствие", 0.5)]:
            for month in months[-n:]:
                rows.append(
                    {
                        "date": month,
                        "territory_id": official_id,
                        "category": cat,
                        "value": int(level * share),
                    }
                )
    # Строки нарочно перемешаны: порядок в архиве организаторов не обязан быть сортированным.
    consumption = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    territories = pd.DataFrame(
        {
            "territory_id": ["79-0001", "718-2594"],
            "official_id": [1, 2594],
            "region_code": ["79", "718"],
            "mo_name": ["городской округ город Майкоп", "городской округ Сургут"],
            "mo_name_short": ["Майкоп", "Сургут"],
            "mo_kind": ["urban_okrug", "urban_okrug"],
            "oktmo": ["79701000", "71876000"],
            "center_lat": [44.6, 61.25],
            "center_lon": [40.1, 73.4],
        }
    )
    return consumption, territories


def test_build_official_panel_keeps_the_panel_contract() -> None:
    consumption, territories = _official()
    panel = build_official_panel(consumption, territories)

    assert len(panel) == len(consumption)
    assert set(panel["unique_id"]) == {
        "79-0001__total",
        "79-0001__food",
        "718-2594__total",
        "718-2594__food",
    }
    assert panel["unique_id"].is_monotonic_increasing, "панель отсортирована по ряду и месяцу"
    assert panel["ds"].min() == pd.Timestamp("2023-01-01") and panel["y"].dtype == float
    short = panel[panel["territory_id"] == "718-2594"]
    assert (short["n_obs"] == 20).all() and not short["is_complete"].any()
    assert panel["admin_break_at"].isna().all(), "официальный id — МО в постоянных границах"
    validate_panel(panel)


def test_build_official_panel_rejects_unknown_territories_and_categories() -> None:
    consumption, territories = _official()
    with pytest.raises(PanelContractError, match="нет в таблице территорий"):
        build_official_panel(consumption, territories[territories["official_id"] == 1])
    with pytest.raises(PanelContractError, match="неизвестные категории"):
        build_official_panel(consumption.assign(category="Одежда"), territories)


def test_static_features_take_name_and_kind_from_the_territory_table() -> None:
    consumption, territories = _official()
    panel = build_official_panel(consumption, territories)
    static = static_features(panel, territories, n_size_groups=2)

    row = static[static["unique_id"] == "718-2594__food"].iloc[0]
    assert row["region_name"] == "Ханты-Мансийский автономный округ — Югра"
    assert row["federal_district"] == "Уральский"
    assert row["mo_kind"] == "urban_okrug" and row["mo_name"] == "городской округ Сургут"
    assert row["official_id"] == 2594 and row["oktmo"] == "71876000"
    assert np.isclose(row["size_level"], np.log(2000.0))


def _sized() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Четыре территории: 1 ровная, 2 удваивает расходы с июля 2023, 3 появляется в 2024 году,
    4 — в июне 2023."""
    months = [f"2023-{m:02d}" for m in range(1, 13)] + [f"2024-{m:02d}" for m in range(1, 13)]
    rows = []
    for official_id, levels in [
        (1, [1000] * 24),
        (2, [2000] * 6 + [4000] * 18),
        (3, [None] * 12 + [9000] * 12),
        (4, [None] * 5 + [5000] * 19),
    ]:
        for month, level in zip(months, levels, strict=True):
            if level is None:
                continue
            for category, share in [("Все категории", 1.0), ("Продовольствие", 0.5)]:
                rows.append(
                    {
                        "date": month,
                        "territory_id": official_id,
                        "category": category,
                        "value": int(level * share),
                    }
                )
    territories = pd.DataFrame(
        {
            "territory_id": ["79-0001", "79-0002", "79-0003", "79-0004"],
            "official_id": [1, 2, 3, 4],
            "region_code": ["79"] * 4,
            "mo_name": ["район А", "район Б", "район В", "район Г"],
            "mo_name_short": ["А", "Б", "В", "Г"],
            "mo_kind": ["municipal_district"] * 4,
            "oktmo": ["79601000", "79602000", "79603000", "79604000"],
        }
    )
    return build_official_panel(pd.DataFrame(rows), territories), territories


def test_territory_size_uses_only_months_up_to_the_given_moment() -> None:
    """Размер территории — постоянный признак модели на все окна проверки, поэтому он
    оценивается по месяцам не позже самого раннего момента прогноза."""
    panel, territories = _sized()
    until = pd.Timestamp("2023-06-01")
    static = static_features(panel, territories, n_size_groups=3, level_until=until)
    size = static.drop_duplicates("territory_id").set_index("territory_id")
    assert np.isclose(size.loc["79-0001", "size_level"], np.log(1000.0))
    assert np.isclose(size.loc["79-0002", "size_level"], np.log(2000.0)), "рост после июня не виден"
    assert np.isclose(size.loc["79-0004", "size_level"], np.log(5000.0)), "сам месяц входит"
    assert size.loc[["79-0001", "79-0002", "79-0004"], "size_group"].tolist() == [1, 2, 3]
    # Медиана, а не среднее: к сентябрю у второй территории шесть месяцев по 2000 и три по 4000.
    by_september = static_features(panel, territories, 3, level_until=pd.Timestamp("2023-09-01"))
    level = by_september.drop_duplicates("territory_id").set_index("territory_id")["size_level"]
    assert np.isclose(level["79-0002"], np.log(2000.0))

    # Тот же результат, если из панели стереть всё после этого момента.
    past = static_features(panel[panel["ds"] <= until], territories, 3, level_until=until)
    columns = ["unique_id", "size_level", "size_group"]
    known = static[static["territory_id"] != "79-0003"][columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(known, past[columns].reset_index(drop=True))


def test_territory_that_starts_later_gets_no_size_instead_of_a_future_one() -> None:
    """Ряд, начавшийся после момента оценки, размера не получает: оценка по его собственным
    поздним месяцам была бы значением из будущего."""
    panel, territories = _sized()
    static = static_features(panel, territories, 3, level_until=pd.Timestamp("2023-06-01"))
    late = static[static["territory_id"] == "79-0003"]
    assert len(late) == 2 and late["size_level"].isna().all() and late["size_group"].isna().all()
    assert static["size_group"].dropna().isin([1, 2, 3]).all(), "группы — у территорий с размером"
    before = static_features(panel, territories, 2, level_until=pd.Timestamp("2023-05-01"))
    fourth = before[before["territory_id"] == "79-0004"]
    assert fourth["size_level"].isna().all(), "в мае четвёртой территории ещё нет"
    # По умолчанию граница — конец первого года панели; правило то же.
    default = static_features(panel, territories, 3)
    by_territory = default.drop_duplicates("territory_id").set_index("territory_id")["size_level"]
    assert np.isclose(by_territory["79-0002"], np.log(3000.0)), "медиана 2023 года: 2000 и 4000"
    assert np.isnan(by_territory["79-0003"])
