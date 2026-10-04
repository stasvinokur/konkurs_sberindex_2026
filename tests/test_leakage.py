"""Сводные проверки на утечку будущего.

Каждая проверка отвечает на один вопрос: может ли значение, недоступное на момент прогноза,
повлиять на обучение, признак или тревогу детектора.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sbx.core.changepoint.online import cusum, page_hinkley
from sbx.core.changepoint.residuals import causal_zscores, residual_stream
from sbx.core.features.asof import KNOWN_FUTURE, PAST_ONLY, STATIC, ExogenousBlock
from sbx.core.features.asof import published_at as publication_date
from sbx.core.features.lags import make_features
from sbx.core.features.national import build_national_features, national_as_of
from sbx.core.folds import Fold, split
from sbx.core.news import event_calendar_features, event_features_as_of, monthly_news_features
from sbx.core.seasonality import national_seasonal_index, seasonal_index_as_of

CUTOFF = pd.Timestamp("2024-06-01")


def _panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = pd.date_range("2023-01-01", periods=24, freq="MS")
    panel = pd.DataFrame(
        [
            {
                "unique_id": f"u{i}",
                "ds": d,
                "y": 1000.0 + 10 * t + 100 * i,
                "category": "Все категории",
            }
            for i in range(3)
            for t, d in enumerate(ds)
        ]
    )
    static = pd.DataFrame(
        {
            "unique_id": [f"u{i}" for i in range(3)],
            "category": ["Все категории"] * 3,
            "category_slug": ["total"] * 3,
            "region_code": ["46", "45", "78"],
            "federal_district": ["Центральный"] * 3,
            "mo_kind": ["urban_okrug"] * 3,
            "size_group": [1, 2, 3],
            "size_level": [5.0, 6.0, 7.0],
        }
    )
    seasonal = pd.DataFrame(
        {"category": "Все категории", "month": range(1, 13), "index": np.linspace(0.9, 1.1, 12)}
    )
    return panel, static, seasonal


def test_split_train_never_contains_future() -> None:
    panel, _, _ = _panel()
    fold = Fold("f", CUTOFF, 3, "main")
    train, test = split(panel, fold)
    assert train["ds"].max() == CUTOFF
    assert test["ds"].min() > CUTOFF
    assert test["ds"].max() <= fold.test_end


def test_model_features_do_not_change_when_future_is_deleted() -> None:
    """Признаки на origin должны совпадать при полной панели и при обрезанной по cutoff."""
    panel, static, seasonal = _panel()
    full = make_features(panel, static, seasonal, [CUTOFF], horizon=3, require_target=False)
    truncated = make_features(
        panel[panel["ds"] <= CUTOFF], static, seasonal, [CUTOFF], horizon=3, require_target=False
    )
    columns = [c for c in full.columns if c not in {"y", "target"}]
    pd.testing.assert_frame_equal(
        full[columns].reset_index(drop=True),
        truncated[columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_poisoned_future_row_does_not_affect_training_features() -> None:
    panel, static, seasonal = _panel()
    poisoned = pd.concat(
        [
            panel,
            pd.DataFrame(
                [
                    {
                        "unique_id": "u0",
                        "ds": pd.Timestamp("2024-07-01"),
                        "y": 1e9,
                        "category": "Все категории",
                    }
                ]
            ),
        ]
    )
    fold = Fold("f", CUTOFF, 3, "main")
    train, _ = split(poisoned, fold)
    features = make_features(train, static, seasonal, [CUTOFF], horizon=3, require_target=False)
    assert features.select_dtypes("number").max().max() < 1e8


MONTHS = pd.date_range("2023-01-01", periods=30, freq="MS")


def _blocks() -> list[ExogenousBlock]:
    """Три блока, у каждого значение равно номеру своего месяца: видно, какой месяц взят."""
    number = np.arange(len(MONTHS), dtype=float)
    calendar = pd.DataFrame({"ds": MONTHS, "working_days": 100 + number})
    macro = pd.DataFrame({"origin": MONTHS, "key_rate": 200 + number})
    news = pd.DataFrame(
        {
            "origin": np.tile(MONTHS, 2),
            "region_code": np.repeat(["46", "45"], len(MONTHS)),
            "news_events": np.concatenate([300 + number, 400 + number]),
        }
    )
    return [
        ExogenousBlock("calendar", calendar, KNOWN_FUTURE),
        ExogenousBlock("macro", macro, PAST_ONLY),
        ExogenousBlock("news", news, PAST_ONLY),
    ]


def _number(month: pd.Timestamp) -> float:
    return float(list(MONTHS).index(month))


def test_past_blocks_enter_every_step_as_known_at_the_origin() -> None:
    """Прогноз от июня на три месяца: новости и макро — июньские на всех трёх шагах, а не
    июльские, августовские и сентябрьские; календарь — целевого месяца."""
    panel, static, seasonal = _panel()
    features = make_features(
        panel, static, seasonal, [CUTOFF], horizon=3, require_target=False, exogenous=_blocks()
    )
    assert sorted(features["step"].unique()) == [1, 2, 3]
    origin = _number(CUTOFF)
    assert (features["key_rate"] == 200 + origin).all()
    by_region = features.groupby("region_code", observed=True)["news_events"].unique()
    assert by_region["46"].tolist() == [300 + origin]
    assert by_region["45"].tolist() == [400 + origin]
    assert features.loc[features["region_code"] == "78", "news_events"].isna().all()
    targets = features["ds"].map(_number)
    assert (features["working_days"] == 100 + targets).all(), "календарь — по целевому месяцу"
    assert features["working_days"].nunique() == 3


def test_features_do_not_change_when_blocks_lose_everything_after_the_origin() -> None:
    """Главная проверка на заглядывание: будущее внешних признаков можно стереть целиком, и
    ни один признак прогноза от origin не изменится."""
    panel, static, seasonal = _panel()
    full = _blocks()
    truncated = [
        block
        if block.kind == KNOWN_FUTURE
        else ExogenousBlock(block.name, block.frame[block.frame["origin"] <= CUTOFF], block.kind)
        for block in full
    ]
    kwargs = {"origins": [CUTOFF], "horizon": 3, "require_target": False}
    with_future = make_features(panel, static, seasonal, exogenous=full, **kwargs)
    without_future = make_features(panel, static, seasonal, exogenous=truncated, **kwargs)
    pd.testing.assert_frame_equal(with_future, without_future)
    assert with_future["key_rate"].notna().all() and len(with_future) == 9


def test_training_rows_take_past_blocks_at_their_own_origin() -> None:
    """Обучающая строка с origin апрель получает апрельские значения, а не значения cutoff."""
    panel, static, seasonal = _panel()
    origins = [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-04-01")]
    features = make_features(panel, static, seasonal, origins, horizon=2, exogenous=_blocks())
    assert (features["key_rate"] == 200 + features["origin"].map(_number)).all()


def test_make_features_refuses_untyped_frames_and_clashing_columns() -> None:
    panel, static, seasonal = _panel()
    untyped = pd.DataFrame({"ds": MONTHS, "key_rate": 1.0})
    with pytest.raises(TypeError, match="ExogenousBlock"):
        make_features(panel, static, seasonal, [CUTOFF], horizon=1, exogenous=[untyped])
    clash = ExogenousBlock("calendar", pd.DataFrame({"ds": MONTHS, "month": 1}), KNOWN_FUTURE)
    with pytest.raises(ValueError, match="month"):
        make_features(panel, static, seasonal, [CUTOFF], horizon=1, exogenous=[clash])
    by_territory = ExogenousBlock(
        "access", pd.DataFrame({"origin": MONTHS, "territory_id": "t", "x": 1.0}), PAST_ONLY
    )
    with pytest.raises(ValueError, match="territory_id"):
        make_features(panel, static, seasonal, [CUTOFF], horizon=1, exogenous=[by_territory])


def test_a_block_can_be_keyed_by_territory_without_the_key_becoming_a_feature() -> None:
    panel, static, seasonal = _panel()
    static = static.assign(territory_id=["t0", "t1", "t2"])
    weather = pd.DataFrame(
        {
            "origin": np.tile(MONTHS, 2),
            "territory_id": np.repeat(["t0", "t1"], len(MONTHS)),
            "temperature": np.repeat([-5.0, 7.0], len(MONTHS)),
        }
    )
    block = ExogenousBlock("weather", weather, PAST_ONLY)
    features = make_features(
        panel, static, seasonal, [CUTOFF], horizon=2, require_target=False, exogenous=[block]
    )
    assert "territory_id" not in features.columns
    got = features.groupby("unique_id")["temperature"].first()
    assert got["u0"] == -5.0 and got["u1"] == 7.0 and np.isnan(got["u2"])


def test_a_static_block_gives_every_row_of_a_series_the_same_value() -> None:
    panel, static, seasonal = _panel()
    static = static.assign(territory_id=["t0", "t1", "t2"])
    access = ExogenousBlock(
        "access",
        pd.DataFrame({"territory_id": ["t0", "t1"], "market_access_log": [3.5, 6.0]}),
        STATIC,
    )
    origins = [pd.Timestamp("2024-02-01"), CUTOFF]
    features = make_features(
        panel, static, seasonal, origins, horizon=3, require_target=False, exogenous=[access]
    )
    assert "territory_id" not in features.columns
    by_series = features.groupby("unique_id")["market_access_log"].agg(["nunique", "first"])
    assert by_series.loc["u0"].tolist() == [1, 3.5] and by_series.loc["u1"].tolist() == [1, 6.0]
    assert features.loc[features["unique_id"] == "u2", "market_access_log"].isna().all()
    assert len(features) == 3 * 2 * 3


NATIONAL_LAGS = {"consumper-spending-index-sa": 35, "real-key-interest-rate": 7}


def _national_sources() -> dict[str, pd.DataFrame]:
    """Месячный ряд и недельный ряд; значение равно номеру месяца наблюдения."""
    monthly = pd.DataFrame(
        {"period": MONTHS, "type": "Всего", "value": [_number(m) for m in MONTHS]}
    )
    weeks = pd.date_range(MONTHS[0], MONTHS[-1] + pd.offsets.MonthEnd(0), freq="W-MON")
    weekly = pd.DataFrame(
        {
            "period": weeks,
            "key_rate_categories": "ставка",
            "value": [_number(w.to_period("M").to_timestamp()) for w in weeks],
        }
    )
    return {"consumper-spending-index-sa": monthly, "real-key-interest-rate": weekly}


def test_national_row_at_the_origin_holds_the_latest_published_values() -> None:
    row = national_as_of(build_national_features(_national_sources(), NATIONAL_LAGS), [CUTOFF])
    assert row["origin"].tolist() == [CUTOFF]
    june = _number(CUTOFF)
    # Месячный ряд с лагом 35 дней: к концу июня вышел апрель. Недельный с лагом 7: вышел май.
    assert row["nat35__nat_index_sa_vsego"].iloc[0] == june - 2
    assert row["nat7__key_rate_stavka"].iloc[0] == june - 1


def test_national_row_at_the_origin_ignores_everything_published_later() -> None:
    """Из сырых рядов стирается всё, что вышло после конца месяца origin: строка признаков на
    origin обязана остаться прежней."""
    sources = _national_sources()
    moment = CUTOFF + pd.offsets.MonthEnd(0)
    monthly = sources["consumper-spending-index-sa"]
    weekly = sources["real-key-interest-rate"]
    known = {
        "consumper-spending-index-sa": monthly[publication_date(monthly["period"], 35) <= moment],
        "real-key-interest-rate": weekly[weekly["period"] + pd.Timedelta(days=7) <= moment],
    }
    assert len(known["real-key-interest-rate"]) < len(weekly)
    full = national_as_of(build_national_features(sources, NATIONAL_LAGS), [CUTOFF])
    truncated = national_as_of(build_national_features(known, NATIONAL_LAGS), [CUTOFF])
    pd.testing.assert_frame_equal(full, truncated)


def test_news_features_ignore_events_added_after_cutoff() -> None:
    events = pd.DataFrame(
        {
            "adm1": ["RS48"] * 3,
            "region_code": ["45"] * 3,
            "event_root_code": ["14", "14", "19"],
            "quad_class": [3, 3, 4],
            "goldstein": [-2.0, -3.0, -8.0],
            "num_articles": [5, 6, 7],
            "avg_tone": [-1.0, -2.0, -5.0],
            "date_added": pd.to_datetime(["2024-05-10", "2024-06-20", "2024-07-05"]),
        }
    )
    visible = monthly_news_features(events, cutoff=CUTOFF)
    assert visible["news_events"].sum() == 1
    assert pd.Timestamp("2024-07-01") not in set(visible["ds"])


def test_event_calendar_ignores_events_published_after_cutoff() -> None:
    events = [
        {"date": "2024-07-01", "published_at": "2024-07-05", "region_code": "46", "kind": "flood"},
        {"date": "2024-05-01", "published_at": "2024-05-03", "region_code": "46", "kind": "flood"},
    ]
    months = pd.date_range("2024-05-01", periods=4, freq="MS")
    out = event_calendar_features(events, months, ["46"], cutoff=CUTOFF)
    assert out.set_index("ds").loc[pd.Timestamp("2024-05-01"), "event_flood"] == 1.0
    assert out.set_index("ds").loc[pd.Timestamp("2024-07-01"), "event_flood"] == 0.0


def _national_spending() -> pd.DataFrame:
    """Национальный ряд 2022–2025: с 2023 года декабрьский пик становится вдвое выше.

    К июню 2024 декабрь 2023 уже опубликован, но в оценку индекса ещё не входит: отношению
    к центрированному среднему нужны полгода данных после месяца. Индекс по всему ряду видит
    два высоких декабря из трёх.
    """
    months = pd.date_range("2022-01-01", "2025-12-01", freq="MS")
    shape = np.array([0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.3])
    values = [
        1000.0 * shape[m.month - 1] * (2.0 if m.month == 12 and m.year >= 2023 else 1.0)
        for m in months
    ]
    return pd.DataFrame({"period": months, "type": "Всего", "value": values})


def test_seasonal_index_of_a_fold_ignores_national_data_published_later() -> None:
    """Сезонный индекс окна оценивается по национальному ряду, опубликованному к моменту
    прогноза: будущие месяцы можно стереть, и индекс не изменится."""
    national = _national_spending()
    moment = CUTOFF + pd.offsets.MonthEnd(0)
    known = national[publication_date(national["period"], 35) <= moment]
    assert known["period"].max() == pd.Timestamp("2024-04-01")

    at_origin = seasonal_index_as_of(national, CUTOFF, lag_days=35)
    pd.testing.assert_frame_equal(at_origin, seasonal_index_as_of(known, CUTOFF, lag_days=35))

    whole = national_seasonal_index(national, end=pd.Timestamp("2025-12-01"))
    december = lambda frame: float(frame.loc[frame["month"] == 12, "index"].iloc[0])  # noqa: E731
    assert december(whole) > december(at_origin) + 0.02, (
        "индекс по всему ряду знает о будущих декабрях — тест обязан это различать"
    )


FLOODS = [
    {"date": "2024-05-20", "published_at": "2024-05-20", "region_code": "46", "kind": "flood"},
    # Случилось в июне, а известно стало только в июле.
    {"date": "2024-06-28", "published_at": "2024-07-03", "region_code": "46", "kind": "flood"},
]


def test_event_row_of_an_origin_holds_only_what_was_published_by_its_end() -> None:
    origins = pd.date_range("2024-05-01", periods=4, freq="MS")
    out = event_features_as_of(FLOODS, origins, ["46", "45"]).set_index(["region_code", "origin"])
    june, july = pd.Timestamp("2024-06-01"), pd.Timestamp("2024-07-01")
    assert out.loc[("46", pd.Timestamp("2024-05-01")), "event_flood"] == 1.0
    assert out.loc[("46", june), "event_flood"] == 0.0, "июньское событие ещё не опубликовано"
    assert out.loc[("46", june), "event_flood_decay"] == pytest.approx(0.5), "след майского"
    assert out.loc[("46", july), "event_flood_decay"] == pytest.approx(0.5), "след июньского"
    assert (out.loc["45"] == 0.0).all().all(), "в другом регионе событий нет"


def test_event_row_of_an_origin_ignores_events_published_later() -> None:
    origins = [CUTOFF]
    known = [
        e for e in FLOODS if pd.Timestamp(e["published_at"]) <= CUTOFF + pd.offsets.MonthEnd(0)
    ]
    assert len(known) == 1
    pd.testing.assert_frame_equal(
        event_features_as_of(FLOODS, origins, ["46"]),
        event_features_as_of(known, origins, ["46"]),
    )


@pytest.mark.parametrize("detector", [cusum, page_hinkley])
def test_online_detectors_are_causal(detector) -> None:
    """Тревоги на префиксе ряда совпадают с тревогами на полном ряду."""
    rng = np.random.default_rng(0)
    signal = np.concatenate([rng.normal(size=40), rng.normal(loc=4.0, size=40)])
    cut = 50
    full = [a for a in detector(signal, threshold=6.0) if a < cut]
    prefix = detector(signal[:cut], threshold=6.0)
    assert full == prefix


def _shifted_series(n: int = 24, tau: int = 15) -> pd.DataFrame:
    """Два ряда с шумом; в первом с позиции `tau` уровень вырастает на 40%."""
    rng = np.random.default_rng(7)
    months = pd.date_range("2023-01-01", periods=n, freq="MS")
    frames = []
    for uid, shift in (("synth_0", 1.4), ("synth_1", 1.0)):
        values = 1000.0 + rng.normal(0, 15, size=n)
        values[tau:] *= shift
        frames.append(
            pd.DataFrame({"unique_id": uid, "ds": months, "y": values, "category": "Все категории"})
        )
    return pd.concat(frames, ignore_index=True)


FLAT = pd.DataFrame({"category": "Все категории", "month": range(1, 13), "index": 1.0})


def test_causal_zscores_use_strictly_past_values() -> None:
    values = np.array([10.0, 12.0, 11.0, 13.0, 50.0, 12.0])
    z = causal_zscores(values, min_history=3)
    assert np.isnan(z[:3]).all(), "пока истории мало, оценки масштаба нет"
    past = values[:4]
    scale = 1.4826 * np.median(np.abs(past - np.median(past)))
    assert z[4] == pytest.approx((50.0 - np.median(past)) / scale), (
        "выброс не входит в свой масштаб"
    )
    np.testing.assert_allclose(causal_zscores(values[:5], min_history=3), z[:5], equal_nan=True)
    uncentered = causal_zscores(values, center=False, min_history=3)
    assert uncentered[4] == pytest.approx(50.0 / scale)


def test_residual_stream_on_a_prefix_equals_the_stream_on_the_full_series() -> None:
    """Поток, который видят детекторы, не должен зависеть от будущих значений ряда: сдвиг в
    конце ряда не меняет нормированные остатки его начала."""
    series = _shifted_series()
    cut = series["ds"].sort_values().unique()[11]
    full = residual_stream(series, FLAT)
    prefix = residual_stream(series[series["ds"] <= cut], FLAT)
    head = full[full["ds"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(head, prefix.reset_index(drop=True))
    assert head["z"].notna().sum() > 10
    shifted = full[full["unique_id"] == "synth_0"].reset_index(drop=True)
    assert abs(shifted.loc[15, "z"]) > 5, "сам сдвиг в потоке виден"
    # Остаток не центрируется: систематическое смещение прогноза — тоже сигнал. Масштаб — по
    # остаткам до этого месяца.
    past = shifted.loc[1:14, "residual"].to_numpy()
    scale = 1.4826 * np.median(np.abs(past - np.median(past)))
    assert shifted.loc[15, "z"] == pytest.approx(shifted.loc[15, "residual"] / scale)
