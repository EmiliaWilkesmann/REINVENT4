from reinvent.runmodes.RL.scaffold_pruner import ScaffoldPruner
from reinvent.chemistry.library_design.attachment_points import remove_attachment_point_numbers


def test_prunes_bad_variant():
    p = ScaffoldPruner(["A", "B", "C"], warmup=0, margin=0.2, alpha=1.0)
    changed = p.update(["A", "B", "C"], [0.9, 0.85, 0.1], [True] * 3, step=1)
    assert changed
    assert p.active == ["A", "B"]


def test_invalid_samples_ignored():
    p = ScaffoldPruner(["A", "B"], warmup=0, margin=0.2, alpha=1.0)
    changed = p.update(["A", "B"], [0.9, 0.0], [True, False], step=1)
    assert not changed
    assert p.active == ["A", "B"]      # B hat keinen gültigen Wert, bleibt


def test_warmup_blocks_pruning():
    p = ScaffoldPruner(["A", "B"], warmup=10, margin=0.2, alpha=1.0)
    assert not p.update(["A", "B"], [0.9, 0.1], [True, True], step=3)
    assert p.active == ["A", "B"]


def test_normalize_maps_back_to_original_lines():
    lines = ["[*:0]CC[*:1]", "[*:0]CCC[*:1]"]
    p = ScaffoldPruner(lines, normalize=remove_attachment_point_numbers,
                       warmup=0, margin=0.2, alpha=1.0)
    p.update(["[*]CC[*]", "[*]CCC[*]"], [0.9, 0.1], [True, True], step=1)
    assert p.active == ["[*:0]CC[*:1]"]   # Originalzeile mit Nummern bleibt erhalten