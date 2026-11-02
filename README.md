# MovePort Cross-Condition EMG Classification

This repository contains the original cross-condition EMG classification experiment extracted verbatim from the supplied notebook text.

## Source preservation

- Source: `Pasted text(20261102-214531).txt`
- Original source lines: 2006–2973 (inclusive)
- The Python code has not been rewritten, refactored, optimized, or methodologically changed.
- Data-loading behavior, feature calculations, window size, conditions, subject selection, scaling, classifier, LOSO procedure, metrics, plotting, and output paths remain exactly as in the original source.

## Run environment

The source uses Google Colab-style paths rooted at `/content/MovePort` and imports:

- numpy
- pandas
- matplotlib
- scikit-learn

The original script expects the MovePort dataset at the path configured in the script.

## Excluded material

Other notebook sections include earlier/alternate feature extraction, Subject 25 processing, separate window-count scripts, hardcoded supplementary correlation calculations, directory inspection, and exploratory signal plots. They are not included in this primary experiment file. No excluded code has been silently merged into or substituted for the primary experiment.

Keep the original notebook/text export as the complete research record.
