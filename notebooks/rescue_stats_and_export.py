# =====================================================================
# RESCUE — turn perturbation pickles already on Drive into the
# {GENE}_stats.csv files Benchmate needs.
#
# Your runtime was recycled: Drive isn't mounted and geneformer is gone.
# The pickles are safe on Drive, so you do NOT need to re-run the GPU
# perturbation. Paste STEP 1 into a cell, run it, wait for the restart,
# then paste STEP 2 into a new cell.
# =====================================================================


# ---------------------------------------------------------------------
# STEP 1 — mount Drive, find the pickles, reinstall geneformer
# ---------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

import glob, os

# find whatever folder the perturbation actually wrote to
cands = sorted(glob.glob('/content/drive/MyDrive/*/perturbations'))
print('perturbation folders on Drive:')
for c in cands:
    genes = [os.path.basename(g) for g in sorted(glob.glob(f'{c}/*'))
             if os.path.isdir(g)]
    print(f'  {c}')
    for g in genes:
        n = len(glob.glob(f'{c}/{g}/*'))
        print(f'      {g:>10}  {n} file(s)')
if not cands:
    print('  NONE FOUND — check My Drive in the file browser on the left.')

# reinstall the stack (same logic as the notebook's setup cell: never let
# pip move numpy, and hold transformers on 4.x for SpecialTokensMixin)
def _deps_ready():
    try:
        import numpy, pandas, geneformer  # noqa: F401
        return True
    except Exception:
        return False

if _deps_ready():
    print('\ndeps already installed — skip to STEP 2.')
else:
    import numpy, subprocess, sys
    with open('/tmp/constraints.txt', 'w') as f:
        f.write(f'numpy=={numpy.__version__}\n')
    print(f'\nholding numpy at {numpy.__version__}')
    pkgs = ['transformers>=4.44,<5', 'tokenizers', 'peft', 'accelerate',
            'datasets<4', 'loompy', 'tdigest', 'anndata', 'pyarrow',
            'statsmodels']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    '--no-cache-dir', '-c', '/tmp/constraints.txt', *pkgs],
                   check=False)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    '--no-cache-dir', '--no-deps',
                    'git+https://huggingface.co/ctheodoris/Geneformer'],
                   check=False)
    print('Installed. RESTARTING the runtime — Colab will say the session')
    print('crashed. That is expected. Drive stays mounted. Then run STEP 2.')
    import IPython
    IPython.Application.instance().kernel.do_shutdown(True)


# =====================================================================
# STEP 2 — aggregate + export   (paste into a NEW cell after the restart)
# =====================================================================
import os, glob, shutil
import pandas as pd

# `datasets` imports torchvision's VideoReader when it thinks torchvision
# is installed; current torchvision dropped it.
try:
    import datasets.config
    datasets.config.TORCHVISION_AVAILABLE = False
except Exception:
    pass

# auto-detect the folder STEP 1 printed; override by hand if you like
_cands = sorted(glob.glob('/content/drive/MyDrive/*/perturbations'))
PERTURB_OUT = _cands[0] if _cands else None
assert PERTURB_OUT, 'No perturbations folder on Drive — is Drive mounted?'
print(f'using {PERTURB_OUT}')

SYMBOLS = [os.path.basename(d) for d in sorted(glob.glob(f'{PERTURB_OUT}/*'))
           if os.path.isdir(d)]
print(f'genes: {SYMBOLS}')

# ---- aggregate the pickles into one CSV per gene ---------------------
from geneformer import InSilicoPerturberStats

def run_stats(sym):
    in_dir = f'{PERTURB_OUT}/{sym}'
    stats = InSilicoPerturberStats(
        mode='aggregate_gene_shifts',
        genes_perturbed='all',
        combos=0,
        anchor_gene=None,
        cell_states_to_model=None,
    )
    stats.get_stats(
        input_data_directory=in_dir,
        null_dist_data_directory=None,
        output_directory=in_dir,
        output_prefix=f'{sym}_stats',
    )

for sym in SYMBOLS:
    print(f'\nAggregating {sym}...')
    try:
        run_stats(sym)
    except Exception as e:
        print(f'  FAILED: {e}')

# ---- validate + export for Benchmate --------------------------------
EXPORT_DIR = '/content/benchmate_export'
os.makedirs(EXPORT_DIR, exist_ok=True)
REQUIRED = ['Affected_Ensembl_ID', 'Cosine_sim_mean']

def find_stats_csv(sym):
    exact = f'{PERTURB_OUT}/{sym}/{sym}_stats.csv'
    if os.path.exists(exact):
        return exact
    hits = glob.glob(f'{PERTURB_OUT}/{sym}/*.csv')
    return hits[0] if hits else None

ready, problems = [], []
for sym in SYMBOLS:
    src = find_stats_csv(sym)
    if not src:
        problems.append((sym, 'no CSV — stats step did not finish'))
        continue
    df = pd.read_csv(src)
    if 'Affected' in df.columns:
        df = df[df['Affected'] != 'cell_emb']
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        problems.append((sym, f'missing columns {missing}'))
        continue
    # the runtime restarted, so `adata` is gone — leave Ensembl IDs in the
    # symbol column. Benchmate only requires the two columns above.
    if 'Affected_gene_name' not in df.columns:
        df['Affected_gene_name'] = df['Affected_Ensembl_ID']
    if 'N_Detections' not in df.columns:
        df['N_Detections'] = 0
    df = df.dropna(subset=['Affected_Ensembl_ID'])
    df = df.sort_values('Cosine_sim_mean', ascending=True)
    out = f'{EXPORT_DIR}/{sym}_stats.csv'
    df.to_csv(out, index=False)
    ready.append((sym, out, len(df)))

print('\nReady for Benchmate')
print('-' * 52)
for sym, out, n in ready:
    print(f'  {sym:>10}_stats.csv   {n:>5} rows')
for sym, why in problems:
    print(f'  {sym:>10}  SKIPPED — {why}')

# keep a copy on Drive
try:
    dst = '/content/drive/MyDrive/benchmate_export'
    os.makedirs(dst, exist_ok=True)
    for _, out, _ in ready:
        shutil.copy(out, dst)
    print(f'\nAlso copied to {dst}')
except Exception:
    pass

# download
try:
    from google.colab import files
    import time
    for _, out, _ in ready:
        files.download(out)
        time.sleep(1.5)
    print('\nDownloads triggered.')
except Exception as e:
    print(f'\n(Auto-download unavailable: {e}) — files are in {EXPORT_DIR}')

print('\nNext: Benchmate -> sidebar -> Upload CSVs. Keep the filenames as-is.')
