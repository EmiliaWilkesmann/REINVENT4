import numpy as np


class ScaffoldPruner:
    """Entfernt Scaffold-Varianten, deren gleitender Score deutlich unter der besten liegt."""

    def __init__(self, scaffolds, normalize=None, alpha=0.1, warmup=50, margin=0.2):
        self.normalize = normalize or (lambda s: s)
        self.alpha = alpha
        self.warmup = warmup
        self.margin = margin
        self.active = list(scaffolds)   # Originalzeilen der .smi, Reihenfolge bleibt
        self.ema = {}                   # normalisierter Scaffold -> gleitender Mittelwert
        self.history = []               # (step, entfernte Zeilen)

    def update(self, sampled_scaffolds, scores, valid_mask, step):
        """Gibt True zurück, wenn sich die aktive Liste geändert hat."""
        per_scaffold = {}
        for scaf, sc, ok in zip(sampled_scaffolds, scores, valid_mask):
            if ok and not np.isnan(sc):
                per_scaffold.setdefault(self.normalize(scaf), []).append(sc)

        for key, vals in per_scaffold.items():
            m = float(np.mean(vals))
            old = self.ema.get(key, m)
            self.ema[key] = (1 - self.alpha) * old + self.alpha * m

        if step < self.warmup:
            return False

        known = {}
        for line in self.active:
            key = self.normalize(line)
            if key in self.ema:
                known[line] = self.ema[key]
        if len(known) < 2:
            return False

        best = max(known.values())
        keep = [l for l in self.active if l not in known or known[l] >= best - self.margin]
        if len(keep) == len(self.active):
            return False

        self.history.append((step, [l for l in self.active if l not in keep]))
        self.active = keep
        return True