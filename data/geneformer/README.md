# Geneformer perturbation cache

This directory holds the pre-computed in-silico perturbation results that the
Co-Scientist agents read from. Each file is named `{GENE_SYMBOL}_stats.csv`
and is produced by `InSilicoPerturberStats` in `notebooks/02_geneformer_ciliated_cells.ipynb`.

## Expected schema

The agents look for these columns (extra columns are ignored):

| column | meaning |
|---|---|
| `Affected` | "cell_emb" or an Ensembl ID. Rows with "cell_emb" are dropped. |
| `Affected_Ensembl_ID` | Ensembl ID of the affected gene. Required. |
| `Affected_gene_name` | HGNC symbol of the affected gene. |
| `Cosine_sim_mean` | Mean cosine similarity between perturbed and baseline embedding. Lower = bigger predicted shift. |
| `N_Detections` | Number of cells in which the shift was measured. Used to filter low-support hits. |

## How to populate

After running `notebooks/02_geneformer_ciliated_cells.ipynb` in Colab, the
stats CSVs are saved to your Google Drive at:

```
MyDrive/benchmate_geneformer_cilia/perturbations/{GENE}/{GENE}_stats.csv
```

Download each file (Google Drive web → right-click → Download, or via Drive
desktop app sync) and place them here:

```
data/geneformer/
├── TXNDC15_stats.csv
├── SYVN1_stats.csv
└── MARCHF6_stats.csv
```

## How it's used

The Generation and Reflection agents scan every research goal / hypothesis
for gene symbols. If a mentioned gene has a `*_stats.csv` in this directory,
the top 10 affected genes (filtered by `N_Detections ≥ 200`, sorted by
biggest shift) are injected into the agent's prompt as ground-truth
perturbation evidence.

To check what's available from Python:

```python
from co_scientist.tools import available_geneformer_genes, geneformer_neighbors
print(available_geneformer_genes())
print(geneformer_neighbors("TXNDC15", top_n=5))
```

## Adding more perturbations

Run notebook 02 with a different `TARGETS` dict (any gene with a valid
Ensembl ID will work). Drop the resulting `_stats.csv` into this directory
and the agents will pick it up automatically — no code changes needed.
