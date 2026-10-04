import pandas as pd

from sbx.core.gdelt_archive import (
    GdeltManifest,
    MonthSpec,
    archive_problems,
    coverage_problems,
    manifest_problems,
    month_days,
    month_file,
    months_between,
)


def _manifest(months: tuple[MonthSpec, ...], missing: tuple[str, ...] = ()) -> GdeltManifest:
    return GdeltManifest(
        source="GDELT",
        url_template="https://x/{date:%Y%m%d}.zip",
        country="RS",
        start=months[0].month,
        end=months[-1].month,
        missing_days=missing,
        months=months,
    )


def _spec(month: str, sha: str = "a" * 64, rows: int = 10) -> MonthSpec:
    return MonthSpec(month, month_file(month, "RS"), sha, rows)


def test_months_between_covers_the_period_across_a_year_boundary() -> None:
    assert months_between("2022-11", "2023-02") == ["2022-11", "2022-12", "2023-01", "2023-02"]
    assert months_between("2024-05", "2024-05") == ["2024-05"]


def test_month_file_keeps_the_name_the_archive_already_uses() -> None:
    assert month_file("2022-11", "RS") == "external/gdelt/gdelt_rs_202211.parquet"


def test_month_days_lists_every_calendar_day() -> None:
    days = month_days("2024-02")
    assert len(days) == 29, "февраль високосного года"
    assert days[0] == pd.Timestamp("2024-02-01")
    assert days[-1] == pd.Timestamp("2024-02-29")


def test_month_days_leaves_out_days_the_source_never_published() -> None:
    """За эти дни у GDELT нет суточного файла: запрашивать их незачем, а архив без них полон."""
    days = month_days("2022-11", missing_days=("2022-11-10", "2023-03-23"))
    assert len(days) == 29
    assert pd.Timestamp("2022-11-10") not in days


def test_coverage_problems_name_a_day_lost_during_download() -> None:
    """Каждый суточный файл несёт события со своей датой добавления: по ним виден пропавший день."""
    missing = ("2022-11-10",)
    full = month_days("2022-11", missing)
    assert coverage_problems("2022-11", full, missing) == []

    lost = [d for d in full if d != pd.Timestamp("2022-11-21")]
    assert coverage_problems("2022-11", lost, missing) == ["2022-11: нет событий за 2022-11-21"]

    stray = [*full, pd.Timestamp("2022-12-01")]
    assert "2022-12-01" in coverage_problems("2022-11", stray, missing)[0]


def test_archive_problems_names_absent_and_mismatching_months() -> None:
    manifest = _manifest((_spec("2024-01"), _spec("2024-02"), _spec("2024-03", rows=7)))
    actual = {
        month_file("2024-01", "RS"): ("a" * 64, 10),
        # Февраля нет вовсе, март недокачан: строк меньше и другая контрольная сумма.
        month_file("2024-03", "RS"): ("b" * 64, 5),
    }
    problems = archive_problems(manifest, actual)
    assert len(problems) == 3
    assert "gdelt_rs_202402.parquet: файл отсутствует" in problems[0]
    assert "gdelt_rs_202403.parquet: sha256 не совпадает" in problems[1]
    assert "строк 5, ожидалось 7" in problems[2]

    complete = {spec.path: (spec.sha256, spec.rows) for spec in manifest.months}
    assert archive_problems(manifest, complete) == []


def test_manifest_problems_catch_a_gap_and_a_stray_missing_day() -> None:
    good = _manifest((_spec("2024-01"), _spec("2024-02")), missing=("2024-01-15",))
    assert manifest_problems(good) == []

    gap = GdeltManifest(
        "GDELT", "u", "RS", "2024-01", "2024-03", (), (_spec("2024-01"), _spec("2024-03"))
    )
    assert any("2024-02" in p for p in manifest_problems(gap))

    stray = _manifest((_spec("2024-01"),), missing=("2023-12-31",))
    assert any("2023-12-31" in p for p in manifest_problems(stray))
