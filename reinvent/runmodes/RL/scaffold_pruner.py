"""Scaffold-Pruning für LibInvent-RL.

Der Pruner merkt sich pro Scaffold einen gleitenden Score (EMA) und entscheidet
nach einer Warm-up-Phase, welche Zeile der Scaffold-Liste entfernt wird.
Er kennt weder REINVENT noch Torch, nur Listen und Zahlen, und lässt sich
deshalb mit erfundenen Daten testen.

Feste Regeln:
- Pro Schritt wird höchstens EINE Zeile entfernt (die schlechteste).
- Nach jeder Entfernung gibt es eine Pause (cooldown), in der nichts bewertet wird.
"""

import csv

import numpy as np


class ScaffoldPruner:
    """Entfernt Scaffold-Varianten, die über mehrere Schritte deutlich unter der besten liegen.

    scaffolds:    Originalzeilen der .smi (mit Nummern an den Anknüpfungspunkten)
    normalize:    Funktion, die Strings in eine Einheitsform bringt
                  (z. B. Nummern entfernen), damit Proben und Zeilen zusammenpassen
    alpha:        Gewicht des neuesten Schritts im gleitenden Mittelwert (0 < alpha <= 1)
    warmup:       vor diesem Schritt wird nichts entfernt
    cooldown:     nach einer Entfernung so viele Schritte Pause, bevor wieder bewertet wird;
                  None = gleich lang wie warmup
    margin:       erlaubter Abstand zum besten Scaffold (>= 0)
    patience:     so viele bewertete Schritte hintereinander muss ein Scaffold unter der
                  Schwelle liegen, bevor er entfernt wird (1 = sofort)
    min_keep:     Untergrenze für die Zahl aktiver Zeilen; ein Schutz, kein Zwang
    min_samples:  so viele gültige Proben insgesamt, bevor ein Scaffold bewertet wird
    aggregate:    "mean" (Mittel aller gültigen Proben eines Schritts) oder
                  "top" (Mittel der besten Proben eines Schritts)
    top_fraction: Anteil der besten Proben bei aggregate="top"
    samples_csv:  Pfad für ein Protokoll aller Proben (None = kein Protokoll);
                  eine bestehende Datei wird beim Start überschrieben
    """

    def __init__(
        self,
        scaffolds,
        normalize=None,
        alpha=0.1,
        warmup=50,
        margin=0.2,
        patience=1,
        min_keep=1,
        min_samples=1,
        aggregate="mean",
        top_fraction=0.25,
        samples_csv=None,
        cooldown=None,
    ):
        if not 0 < alpha <= 1:
            raise ValueError("alpha muss in (0, 1] liegen")
        if margin < 0:
            raise ValueError("margin darf nicht negativ sein")
        if patience < 1:
            raise ValueError("patience muss mindestens 1 sein")
        if min_keep < 1:
            raise ValueError("min_keep muss mindestens 1 sein")
        if aggregate not in ("mean", "top"):
            raise ValueError("aggregate muss 'mean' oder 'top' sein")
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction muss in (0, 1] liegen")
        if cooldown is not None and cooldown < 0:
            raise ValueError("cooldown darf nicht negativ sein")

        self.normalize = normalize or (lambda s: s)
        self.alpha = alpha
        self.warmup = warmup
        self.cooldown = warmup if cooldown is None else cooldown
        self.margin = margin
        self.patience = patience
        self.min_keep = min_keep
        self.min_samples = min_samples
        self.aggregate = aggregate
        self.top_fraction = top_fraction
        self.samples_csv = samples_csv

        self.active = list(scaffolds)  # create a list (called self.active) with the input of the scaffolds-list, the order of the scaffolds is preserved, to avoid deleting wrong scaffolds in the original list
        self.ema = {}  # normalisierter Scaffold -> gleitender Wert
        self.n_samples = {}  # normalisierter Scaffold -> Zahl gültiger Proben insgesamt
        self.below_streak = {}  # Originalzeile -> bewertete Schritte in Folge unter der Schwelle
        self.last_removal_step = None  # Schritt der letzten Entfernung
        self.history = []  # (Schritt, entfernte Zeilen)

        if self.samples_csv:
            with open(self.samples_csv, "w", newline="") as f:
                csv.writer(f).writerow(["step", "scaffold", "score", "valid"])

    def _step_value(self, vals):
        """Fasst die gültigen Scores eines Scaffolds in einem Schritt zu einer Zahl zusammen."""
        if self.aggregate == "mean":
            return float(np.mean(vals))
        k = max(1, int(np.ceil(len(vals) * self.top_fraction)))
        return float(np.mean(np.sort(vals)[-k:]))

    def _write_rows(self, rows):
        with open(self.samples_csv, "a", newline="") as f:
            csv.writer(f).writerows(rows)

    def update(self, sampled_scaffolds, scores, valid_mask, step):
        """Verarbeitet einen RL-Schritt. Gibt True zurück, wenn eine Zeile entfernt wurde."""
        per_scaffold = {}
        rows = []

        # 1) Jede Probe gehört zu dem Scaffold, aus dem sie entstanden ist.
        for scaf, sc, ok in zip(sampled_scaffolds, scores, valid_mask):
            key = self.normalize(scaf)
            good = bool(ok) and not np.isnan(sc)
            rows.append((step, key, float(sc), int(good)))
            if good:
                per_scaffold.setdefault(key, []).append(sc)

        if self.samples_csv:
            self._write_rows(rows)

        # 2) Pro Scaffold: Wert dieses Schritts, daraus der gleitende Mittelwert.
        #    Das läuft immer, auch im Warm-up und im Cooldown.
        for key, vals in per_scaffold.items():
            m = self._step_value(vals)
            old = self.ema.get(key, m)
            self.ema[key] = (1 - self.alpha) * old + self.alpha * m
            self.n_samples[key] = self.n_samples.get(key, 0) + len(vals)

        # 3) Keine Entscheidung im Warm-up und nicht direkt nach einer Entfernung.
        if step < self.warmup:
            return False

        if self.last_removal_step is not None and step < self.last_removal_step + self.cooldown:
            return False

        # 4) Nur Zeilen mit genug Messwerten können verglichen werden.
        known = {}
        for line in self.active:
            key = self.normalize(line)
            if key in self.ema and self.n_samples.get(key, 0) >= self.min_samples:
                known[line] = self.ema[key]

        if len(known) < 2:
            return False

        # 5) Wer liegt unter der Schwelle? Die Serie zählt hoch, Erholung setzt sie zurück.
        threshold = max(known.values()) - self.margin
        candidates = []

        for line, value in known.items():
            if value < threshold:
                self.below_streak[line] = self.below_streak.get(line, 0) + 1

                if self.below_streak[line] >= self.patience:
                    candidates.append(line)
            else:
                self.below_streak[line] = 0

        # 6) Höchstens EINE Zeile: die schlechteste, und nie unter min_keep.
        if not candidates or len(self.active) <= self.min_keep:
            return False

        worst = min(candidates, key=lambda line: known[line])

        self.history.append((step, [worst]))
        self.active = [line for line in self.active if line != worst]

        # 7) Neustart: Schwelle und Serien gelten nach der Entfernung nicht mehr.
        self.below_streak.clear()
        self.last_removal_step = step

        return True