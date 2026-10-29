from vlm_bench.ranking import rank_track


def test_anchored_bands_are_non_transitive_and_latency_orders_inside_each_band():
    result = rank_track(
        [
            {"model": "a", "score": 95.0, "median_latency_seconds": 0.2},
            {"model": "b", "score": 94.6, "median_latency_seconds": 0.1},
            {"model": "c", "score": 94.2, "median_latency_seconds": 0.05},
        ],
        accuracy_margin_points=0.5,
    )

    assert [[row["model"] for row in band["models"]] for band in result["bands"]] == [
        ["b", "a"],
        ["c"],
    ]
    assert result["winner"] == "b"
    assert result["ranked_models"][0]["rank"] == 1
    assert result["ranked_models"][2]["rank"] == 3


def test_missing_top_band_latency_and_incomplete_work_withhold_winner():
    rows = [
        {"model": "a", "score": 99, "median_latency_seconds": 0.1},
        {"model": "b", "score": 98.8, "median_latency_seconds": None},
    ]
    timed = rank_track(rows, accuracy_margin_points=0.5)
    incomplete = rank_track([{**rows[0], "terminal_complete": False}], finalizable=True)

    assert timed["winner"] is None
    assert timed["winner_reason"] == "missing_top_band_latency"
    assert incomplete["winner"] is None
    assert incomplete["winner_reason"] == "incomplete_evaluation"
