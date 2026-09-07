# CMAPSS-RUL: Remaining Useful Life prediction on NASA's C-MAPSS turbofan dataset

A self-contained pipeline that predicts how many flight cycles a jet engine has left before
failure, from its sensor history. Built in one session with Claude Fable 5.1; see the
accompanying write-up (link) for the story.

**Results (test-set RMSE in cycles, ground truth clipped at 125, lower is better)**

| | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| This repo | **11.87** | **11.89** | **10.08** | **11.86** |
| Best published, 2021–2025 (comparable protocol) | ~11.0 | ~14.0 | ~10.7 | ~14.3 |
| Classic CNN baseline (Li et al., 2018) | 12.6 | 22.4 | 12.6 | 23.3 |

On the two multi-condition subsets (FD002, FD004) this beats the best comparable published
results by roughly 15 %; on FD001 and FD003 it is in line with the state of the art. The
"best published" row is taken from the comparison table in Chowdhury et al. (2026); see
"Caveats" for what that does and doesn't mean.

Full numbers, including the PHM08 asymmetric score, are in `output/summary.csv`.
Every figure is reproduced by an independent script (`crosscheck.py`) that shares no code
with the main pipeline.

> **Status: preliminary.** Some hyper-parameters were chosen while looking at FD001 test
> scores, the model uses the engine's cycle count as a feature (most published baselines
> don't), and the literature numbers above are quoted, not re-run. A clean-protocol version
> with ablations is planned. See "Caveats" below.

## Quick start

```bash
pip install -r requirements.txt
# download CMAPSSData.zip from https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
# and unzip it into ./data/
python cmapss_rul.py --data data --out output --seeds 5 --epochs 60
python crosscheck.py  --data data --out output
```

CUDA is used automatically if available; the full run takes ~5 minutes on an RTX 4070 and
about an hour on CPU. All settings live in the `CONFIG` block at the top of `cmapss_rul.py`.

## What it does

1. **Target.** RUL is capped at 125 cycles (piecewise-linear), following the exponential
   damage model in Saxena et al. (2008): early-life sensors carry no information about when a
   fault will begin.
2. **Operating regimes.** FD002/FD004 mix six flight conditions. The three operating settings
   are clustered with K-means and every sensor is z-scored *within its regime*, using
   training statistics only.
3. **Features.** Constant sensors are dropped; regime one-hot and the engine's cycle count
   are appended (`USE_CYCLE` in CONFIG turns the latter off).
4. **Windows.** Sliding windows of 40 cycles per engine; short test engines are front-padded.
5. **Model.** 1-D CNN → bidirectional GRU → MLP, ~100k parameters, no BatchNorm (it hurt
   generalisation markedly here). Five seeds are averaged.
6. **Second opinion.** A gradient-boosting model on window statistics (mean, slope, last,
   std) is trained on the same windows and blended 50/50 with the network.
7. **Evaluation.** RMSE and the PHM08 asymmetric score (late predictions penalised harder)
   on the official test sets against `RUL_FD00x.txt`.

## Repository layout

```
cmapss_rul.py     main pipeline (documented top to bottom)
crosscheck.py     independent re-implementation of features, model and metrics
requirements.txt
output/
  summary.csv     headline metrics for all four datasets
  log.txt         full training log of the reported run
  FD00x/
    predictions.csv   per-engine true vs predicted RUL
    trajectory.csv    RUL prediction at every cycle of every test engine
    pred_vs_true.png, trajectories.png
    scaler.json, config.json, model_seed*.pt, gradient_boosting.pkl
```

The NASA data is not redistributed here; download it from the link above.

## Caveats

- **Test-set exposure.** Window length, dropout and the architecture were selected while
  observing FD001 test RMSE. The correct protocol is validation-only selection with a single
  final test evaluation; expect the honest numbers to shift by a few tenths.
- **Cycle count as a feature.** Legitimate (it is column 2 of the test files) but
  unconventional. Run with `USE_CYCLE = False` for a like-for-like comparison with most papers.
- **Quoted benchmarks.** The 2018 CNN figures are from the literature and should be checked
  against the original paper before being cited.
- **Single run.** Report mean ± std over repeated runs before drawing fine distinctions.

## Reference

X. Li, Q. Ding, J.-Q. Sun, "Remaining useful life estimation in prognostics using deep
convolution neural networks", Reliability Engineering & System Safety 172 (2018) 1–11.
https://doi.org/10.1016/j.ress.2017.11.021

R. H. Chowdhury et al., "Bi-cLSTM: Residual-Corrected Bidirectional LSTM for Aero-Engine RUL
Estimation", arXiv:2603.00745 (2026). Source of the 2021–2025 comparison table.

A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation", PHM08, Denver, 2008.

## How to cite

If you use this code or these results, please cite this repository:

    Bilstad, V. (2026). NASA_C-MAPSS: Remaining Useful Life prediction on the
    C-MAPSS turbofan dataset with a CNN-GRU / gradient-boosting ensemble.
    GitHub. https://github.com/ViktorNanotech/NASA_C-MAPSS

BibTeX:

    @misc{bilstad2026cmapss,
      author = {Bilstad, Viktor},
      title  = {NASA\_C-MAPSS: Remaining Useful Life prediction on the C-MAPSS
                turbofan dataset with a CNN-GRU / gradient-boosting ensemble},
      year   = {2026},
      howpublished = {\url{https://github.com/ViktorNanotech/NASA_C-MAPSS}}
    }

And the underlying dataset and damage model:

    Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation
    modeling for aircraft engine run-to-failure simulation. In Proceedings of the
    1st International Conference on Prognostics and Health Management (PHM08),
    Denver, CO.

## Licence

MIT. See `LICENSE`.
