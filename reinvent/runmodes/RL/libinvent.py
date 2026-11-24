"""The LibInvent optimization algorithm"""

from __future__ import annotations

__all__ = ["LibinventLearning", "LibinventTransformerLearning"]
from typing import TYPE_CHECKING

import numpy as np

from .learning import Learning
from reinvent.models.model_factory.sample_batch import SmilesState
from reinvent_plugins.normalizers.rdkit_smiles import normalize

if TYPE_CHECKING:
    from reinvent.scoring import ScoreResults


from .scaffold_pruner import ScaffoldPruner
from reinvent.chemistry.library_design import attachment_points


class LibinventLearning(Learning):
    """LibInvent optimization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_pruning = True  # False = Trockenlauf: nur protokollieren
        self._scaffold_pruner = ScaffoldPruner(
            self.input_smilies,
            #self._state.scaffold_list,
            normalize=attachment_points.remove_attachment_point_numbers,
            warmup=3,
            cooldown=2,
            margin=0.0,
            patience=2,
            alpha=0.5,
            min_keep=1,
            min_samples=3,
            aggregate="mean",
            samples_csv="scaffold_pruner_samples.csv",
            #alpha=self._state.pruning_alpha,
            #warmup=self._state.pruning_warmup,
            #margin=self._state.pruning_margin,
        )

    def _after_score(self, results: ScoreResults, step: int):
        """Hook for additional processing after scoring"""
        changed = self._scaffold_pruner.update(
            self.sampled.items1, results.total_scores, self.invalid_mask, step
        )
        print("EMA", step, self._scaffold_pruner.ema) 
        if changed and self._apply_pruning:
            self.input_smilies = self._scaffold_pruner.active
            print("PRUNE", step, self._scaffold_pruner.active)
    
    def update(self, results: ScoreResults, orig_smilies):
        return self._update_common(results, orig_smilies)

    def score(self):
        """Compute the score for the SMILES strings.

        Overwrites generic method to pass on annotated smiles for use in reaction filter
        """

        connectivity_annotated_smiles = self.sampled.smilies

        results = self.scoring_function(
            normalize(self.sampled.smilies, keep_all=True),
            self.invalid_mask,
            self.duplicate_mask,
            connectivity_annotated_smiles=connectivity_annotated_smiles,
        )

        return results


LibinventTransformerLearning = LibinventLearning
