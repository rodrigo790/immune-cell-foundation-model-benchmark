# Analysis protocol

## Question

Can frozen scGPT embeddings improve lymph node cell annotation across donors?

## Population, analysis and validation

Use the LNCS slice dataset and five author-labelled lineages. Hold out one donor at a time. Compare a fixed 15-marker panel and a scaled PCA-logistic model. In the balanced pilot sample up to 12 cells per lineage and donor, freeze the whole-human encoder, bin within cells and draw up to 127 expressed vocabulary genes per cell. Load all 12 attention blocks strictly into a PyTorch CPU encoder. Fit the linear readout on training donors only.

## Interpretation and scope

The balanced pilot changes class prevalence. The full-data confusion matrix is for PCA-logistic. The pilot measures agreement with reference labels and uses a shorter sequence than pretraining. Overlap with pretraining donors is unknown.

## Primary output

[foundation_pilot.csv](../results/foundation_pilot.csv)

## Source

[Study or method](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE294959)


[Condition-specific recall](../results/condition_recall.csv) gives cell counts and class recall for control and stimulated slices within each held-out donor. The CPU pilot loads the public [scGPT whole-human weights](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo) into a standard PyTorch attention encoder. The [checkpoint manifest](../data/checkpoint.json) records exact input hashes.
