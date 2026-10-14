# Agent notes

## Environment
- Linux only. `scripts/install.sh` builds `.venv` (CUDA torch when `nvidia-smi` works).
- Tests: `.venv/bin/python -m tests.test_chartgen` and `.venv/bin/python -m tests.test_motifs`. There is no linter config.
- Kill background jobs by PID (`ps -eo pid,comm,args | awk '$2 ~ /python/ ...'`). `pkill -f <pattern>` also matches the shell that runs it.

## 6-fret (GHL) part
- Lanes 0-5 are W1 W2 W3 B1 B2 B3 and 7 is open, in the same `(tick, lane, sustain)` tuples as 5-fret. The writer maps lane 5 to `.chart` note 8 (`chart.GHL_NOTE`).
- `chartgen/ghl.py` holds the feature function, `ShapeNet` (Expert shapes), `ReduceNet` (Hard/Medium/Easy keep + shape) and `chart_tiers()`. Weights and per-tier allowed shapes ship in `chartgen/ghl_model.pt`.
- The model shapes the final 5-fret Expert's positions. It does not choose timing.
- Training data are the official FreeStyleGames charts only (Guitar Hero Live on-disc and GHTV, 562 songs). Do not mix in community charts or 5-to-6 converts.
  1. `data/ghl-official/` holds the song folders: Drive rips of the on-disc set, plus `python -m tools.fetch_official_ghl` for the Encore rows with icon `ghl`/`ghtv`.
  2. `python -m tools.build_ghl_dataset data/ghl-official` runs the pipeline's listening front end (Basic Pitch on the mix and on every BS-RoFormer stem) into `data/ghl-features/*.pkl`. It needs about 35 s a song on an RTX 3060 and takes the whole GPU, so run nothing else on the GPU in parallel.
  3. `python -m tools.train_ghl data/ghl-features --sweep 16` trains both models (16 random configurations each, best validation loss wins) and records the configuration and the held-out songs in the model file. Curves: `tensorboard --logdir data/ghl-runs --host 127.0.0.1`.
  4. `python -m tools.eval_ghl -n 10` charts held-out songs end to end and scores the 6-fret part against the official chart.
- The chart-to-audio offset of an official song is its `song.ini` `delay`. Onset matching alone locks onto a beat-multiple on periodic music.
- Official ladder facts (from the official charts): every lower-tier note sits on an Expert tick, Easy uses only W1/W2/W3, W1+W2 and W2+W3, and Medium has no 3-button chords.
