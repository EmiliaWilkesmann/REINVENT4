import csv

import pytest

from reinvent.runmodes.RL.scaffold_pruner import ScaffoldPruner
from reinvent.chemistry.library_design.attachment_points import remove_attachment_point_numbers


def make(lines, **kw):
    """Pruner mit Testwerten: kein Warm-up, nur der neueste Schritt zählt."""
    kw.setdefault("warmup", 0)
    kw.setdefault("margin", 0.2)
    kw.setdefault("alpha", 1.0)
    return ScaffoldPruner(lines, **kw)


# ---------- die vier bisherigen Tests ----------


def test_prunes_bad_variant():
    p = make(["A", "B", "C"])
    changed = p.update(["A", "B", "C"], [0.9, 0.85, 0.1], [True] * 3, step=1)
    assert changed
    assert p.active == ["A", "B"]


def test_invalid_samples_ignored():
    p = make(["A", "B"])
    changed = p.update(["A", "B"], [0.9, 0.0], [True, False], step=1)
    assert not changed
    assert p.active == ["A", "B"]  # B hat keinen gültigen Wert, bleibt


def test_warmup_blocks_pruning():
    p = make(["A", "B"], warmup=10)
    assert not p.update(["A", "B"], [0.9, 0.1], [True, True], step=3)
    assert p.active == ["A", "B"]


def test_normalize_maps_back_to_original_lines():
    lines = ["[*:0]CC[*:1]", "[*:0]CCC[*:1]"]
    p = make(lines, normalize=remove_attachment_point_numbers)
    p.update(["[*]CC[*]", "[*]CCC[*]"], [0.9, 0.1], [True, True], step=1)
    assert p.active == ["[*:0]CC[*:1]"]  # Originalzeile mit Nummern bleibt erhalten


# ---------- neue Tests ----------


def test_two_top_variants_both_kept():
    p = make(["A", "B"])
    assert not p.update(["A", "B"], [0.9, 0.85], [True, True], step=1)
    assert p.active == ["A", "B"]


def test_patience_needs_consecutive_steps():
    p = make(["A", "B"], patience=2)
    assert not p.update(["A", "B"], [0.9, 0.1], [True, True], step=1)  # erste Bewertung darunter
    assert p.active == ["A", "B"]
    assert p.update(["A", "B"], [0.9, 0.1], [True, True], step=2)  # zweite in Folge
    assert p.active == ["A"]


def test_recovery_resets_streak():
    p = make(["A", "B"], patience=2)
    p.update(["A", "B"], [0.9, 0.1], [True, True], step=1)  # Serie: 1
    p.update(["A", "B"], [0.9, 0.9], [True, True], step=2)  # erholt, Serie zurückgesetzt
    assert not p.update(["A", "B"], [0.9, 0.1], [True, True], step=3)  # wieder Serie: 1
    assert p.active == ["A", "B"]


def test_only_one_removal_per_step_worst_first():
    p = make(["A", "B", "C"])
    # B und C liegen beide unter der Schwelle, trotzdem geht nur der schlechteste.
    assert p.update(["A", "B", "C"], [0.9, 0.3, 0.1], [True] * 3, step=1)
    assert p.active == ["A", "B"]
    assert len(p.history) == 1 and p.history[0] == (1, ["C"])


def test_cooldown_blocks_next_removal():
    p = make(["A", "B", "C"], cooldown=3)
    p.update(["A", "B", "C"], [0.9, 0.3, 0.1], [True] * 3, step=1)  # C geht
    assert p.active == ["A", "B"]
    assert not p.update(["A", "B"], [0.9, 0.3], [True] * 2, step=2)  # Pause
    assert not p.update(["A", "B"], [0.9, 0.3], [True] * 2, step=3)  # Pause
    assert p.update(["A", "B"], [0.9, 0.3], [True] * 2, step=4)  # Pause vorbei
    assert p.active == ["A"]


def test_cooldown_defaults_to_warmup():
    p = make(["A", "B", "C"], warmup=5)
    assert p.cooldown == 5
    assert p.update(["A", "B", "C"], [0.9, 0.3, 0.1], [True] * 3, step=5)
    assert not p.update(["A", "B"], [0.9, 0.3], [True] * 2, step=9)
    assert p.update(["A", "B"], [0.9, 0.3], [True] * 2, step=10)
    assert p.active == ["A"]


def test_streak_restarts_after_removal_and_ignores_cooldown_steps():
    p = make(["A", "B", "C"], patience=2, cooldown=2)
    assert not p.update(["A", "B", "C"], [0.9, 0.3, 0.1], [True] * 3, step=1)
    assert p.update(["A", "B", "C"], [0.9, 0.3, 0.1], [True] * 3, step=2)  # C geht
    assert not p.update(["A", "B"], [0.9, 0.3], [True] * 2, step=3)  # Cooldown, zählt nicht
    assert not p.update(["A", "B"], [0.9, 0.3], [True] * 2, step=4)  # Serie von B: 1
    assert p.active == ["A", "B"]
    assert p.update(["A", "B"], [0.9, 0.3], [True] * 2, step=5)  # Serie von B: 2
    assert p.active == ["A"]


def test_min_keep_is_a_lower_bound_not_a_requirement():
    p = make(["A", "B"], min_keep=2)
    assert not p.update(["A", "B"], [0.9, 0.1], [True, True], step=1)
    assert p.active == ["A", "B"]  # B liegt außerhalb des Margins, bleibt wegen min_keep

    q = make(["A", "B"], min_keep=1)
    assert q.update(["A", "B"], [0.9, 0.1], [True, True], step=1)
    assert q.active == ["A"]


def test_min_samples_blocks_judgement():
    p = make(["A", "B"], min_samples=3)
    assert not p.update(["A", "A", "B"], [0.9, 0.9, 0.1], [True] * 3, step=1)  # B: nur 1 Probe
    assert p.active == ["A", "B"]
    p.update(["A", "B", "B"], [0.9, 0.1, 0.1], [True] * 3, step=2)  # B: jetzt 3 Proben
    assert p.active == ["A"]


def test_top_aggregation_protects_good_spikes():
    # A: stabil 0.8. B: im Mittel 0.5, aber eine Probe mit 0.95.
    scaf, sc, ok = ["A", "A", "B", "B"], [0.8, 0.8, 0.95, 0.05], [True] * 4

    mean_pruner = make(["A", "B"], aggregate="mean")
    mean_pruner.update(scaf, sc, ok, step=1)
    assert mean_pruner.active == ["A"]  # Mittel 0.5 liegt unter 0.8 - 0.2

    top_pruner = make(["A", "B"], aggregate="top", top_fraction=0.5)
    top_pruner.update(scaf, sc, ok, step=1)
    assert top_pruner.active == ["A", "B"]  # beste Hälfte von B ergibt 0.95


def test_samples_csv_is_written(tmp_path):
    path = tmp_path / "samples.csv"
    p = make(["A", "B"], samples_csv=str(path))
    p.update(["A", "B"], [0.9, 0.0], [True, False], step=4)

    with open(path) as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["step", "scaffold", "score", "valid"]
    assert rows[1] == ["4", "A", "0.9", "1"]
    assert rows[2] == ["4", "B", "0.0", "0"]


@pytest.mark.parametrize(
    "kw",
    [
        {"alpha": 0},
        {"margin": -0.1},
        {"patience": 0},
        {"min_keep": 0},
        {"aggregate": "median"},
        {"top_fraction": 0},
        {"cooldown": -1},
    ],
)
def test_invalid_settings_raise(kw):
    with pytest.raises(ValueError):
        ScaffoldPruner(["A"], **kw)