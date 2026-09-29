import geopandas as gpd
import pandas as pd
import pytest
import shapely

from src import submarine

_MODELS = (*submarine.BED_MODELS, *submarine.EXCLUDED_BED_MODELS)


def _product(rows: list[tuple[str, int, int, str, float]]) -> gpd.GeoDataFrame:
    """Synthetic product rows: (rgi_id, year_from, year_to, change_type, volume for every model)."""
    frame = pd.DataFrame(
        rows, columns=["rgi_id", "year_from", "year_to", "change_type", "volume"]
    )
    frame["area_m2"] = 1e6
    for model in _MODELS:
        frame[f"{model}_volume_change_m3"] = frame["volume"]
        frame[f"{model}_interpolated_share"] = 0.5
    return gpd.GeoDataFrame(
        frame.drop(columns="volume"),
        geometry=[shapely.box(0, 0, 1, 1)] * len(frame),
        crs=32633,
    )


def test_retreat_is_negative_and_divided_by_whole_coastline_years():
    product = _product(
        [
            ("a", 2012, 2018, "retreat", 6e9),
            ("a", 2012, 2018, "advance", 1.2e9),
            ("b", 2018, 2023, "retreat", 5e9),  # other interval, must be ignored
        ]
    )
    changes = submarine.get_submarine_changes("13_18", product)
    assert changes.dt_yr == 6.0
    assert list(changes.per_glacier.index) == ["a"]
    assert changes.per_glacier.loc["a", "svift_v11"] == pytest.approx(-0.8e9)
    assert changes.per_glacier.loc["a", "retreat_area"] == pytest.approx(1e6)
    assert changes.per_glacier.loc["a", "advance_area"] == pytest.approx(1e6)
    assert submarine.interpolated_share(changes.per_glacier, "frank") == pytest.approx(
        0.5
    )


def test_aggregate_sums_glaciers_before_taking_the_model_spread():
    # Each glacier alone has a half-range of 1; opposite-signed model errors
    # cancel in the total, which a per-glacier quadrature would not capture.
    per_glacier = pd.DataFrame(
        {
            "svift_v11": [-2.0, 0.0],
            "maffezzoli": [0.0, -2.0],
            "frank": [-1.0, -1.0],
            "vanpelt": [-1.0, -1.0],
        }
    )
    assert submarine.aggregate_submarine(per_glacier) == (-2.0, 0.0)


def test_excluded_models_never_enter_the_estimate():
    per_glacier = pd.DataFrame(
        {model: [-1.0] for model in submarine.BED_MODELS}
        | {model: [-100.0] for model in submarine.EXCLUDED_BED_MODELS}
    )
    assert submarine.aggregate_submarine(per_glacier) == (-1.0, 0.0)
    assert "millan" not in submarine.BED_MODELS
    assert "svift_v10" not in submarine.BED_MODELS


def test_model_labels_are_letters_only():
    for model in _MODELS:
        assert submarine.MODEL_LABELS[model].isalpha()
