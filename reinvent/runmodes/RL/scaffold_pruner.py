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
        """Für jeden gesampelten Scaffold wird geschaut: Ist er gültig (ok)? Hat er eine echte Zahl als Score (kein NaN)? Wenn ja → Score in eine Liste pro Scaffold sammeln."""
        for scaf, sc, ok in zip(sampled_scaffolds, scores, valid_mask):
            if ok and not np.isnan(sc):
                per_scaffold.setdefault(self.normalize(scaf), []).append(sc)
    
        """Für jeden Scaffold wird der gleitende Mittelwert (Durchschnitt der neuen Scores) berechnet und in self.ema gespeichert."""
        for key, vals in per_scaffold.items():
            m = float(np.mean(vals))
            old = self.ema.get(key, m)
            self.ema[key] = (1 - self.alpha) * old + self.alpha * m

        """Wenn wir noch in der Warmup-Phase sind, wird keine Pruning-Entscheidung getroffen."""
        if step < self.warmup:
            return False

        """Identifiziert bekannte Scaffold-Varianten"""
        known = {}
        """Man braucht mindestens 2 Scaffolds mit Scores, um überhaupt vergleichen zu können."""
        for line in self.active:
            key = self.normalize(line)
            if key in self.ema:
                known[line] = self.ema[key]
        if len(known) < 2:
            return False

        """Bestimmt den besten Score unter den bekannten Scaffolds und entfernt diejenigen, die mehr als self.margin schlechter sind."""
        best = max(known.values())
        keep = [l for l in self.active if l not in known or known[l] >= best - self.margin]
        if len(keep) == len(self.active):
            return False

        """Speichert die entfernten Scaffolds in der Historie und aktualisiert die aktive Liste.Falls sich was geändert hat: Merke dir, was entfernt wurde, aktualisiere die aktive Liste, gib True zurück."""
        self.history.append((step, [l for l in self.active if l not in keep]))
        self.active = keep
        return True