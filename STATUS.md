# Execution record

Execution date: 2026-11-26. Analysis completed with public research inputs. The exact input bytes are recorded in [the data manifest](data/sources.json). The reproduction commands are in [README.md](README.md).

In a matched pilot of 360 cells from six donors, donor-averaged macro F1 was 0.814 for scGPT, 0.812 for PCA-logistic and 0.728 for markers. With only six donor folds, the 0.002 difference between the learned representations is too small to establish superiority.

The balanced pilot changes class prevalence. The full-data confusion matrix is for PCA-logistic. The pilot measures agreement with reference labels and uses a shorter sequence than pretraining. Overlap with pretraining donors is unknown.

[Repository and current commit history](https://github.com/rodrigo790/immune-cell-foundation-model-benchmark/commits/main) · [Continuous integration](https://github.com/rodrigo790/immune-cell-foundation-model-benchmark/actions)


[Condition-specific recall](results/condition_recall.csv) gives cell counts and class recall for control and stimulated slices within each held-out donor. The CPU pilot loads the public [scGPT whole-human weights](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo) into a standard PyTorch attention encoder. The [checkpoint manifest](data/checkpoint.json) records exact input hashes.
