"""
Statistical modelling sub-package for orthoSynAssign.

Provides four analysis modules that strengthen the HOG-flanking heuristic
with data-driven calibration before deployment on true edge genes:

* :mod:`permutation` — permutation test to confirm HOG-neighbourhood signal
* :mod:`calibrate`   — logistic regression calibration on interior genes
* :mod:`cv`          — genome-stratified cross-validation / ROC analysis
* :mod:`mixed_effects` — mixed-effects logistic regression (R/lme4 wrapper)
* :mod:`apply_model` — apply calibrated threshold to true edge genes
"""
