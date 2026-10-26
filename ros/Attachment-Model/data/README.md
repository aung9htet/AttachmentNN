## Training Data/ Experimental Data

Earlier runs, no robot positions. Labels are as recorded; the controller applied the two parameters the wrong way round (see the repository README).

Training Datas (Recorded 3 minutes per run):
* Index 0: Avoidant 0, Ambivalent 0
* Index 1 - 8: Avoidant 0.1 to 0.8, Ambivalent 0
* Index 9 - 16: Avoidant 0, Ambivalent 0.1 to 0.8

Experimental Datas (Recorded 1 minute per run):
* Index 17: Avoidant 0, Ambivalent 0.0
* Index 18 - 25: Avoidant 0.1 to 0.8, Ambivalent 0
* Index 26 - 33: Avoidant 0, Ambivalent 0.1 to 0.8

## Runs with robot paths

Each folder also has `meta.json`; the data files include odometry paths, true positions and inter-robot distance.

Paper regimes (Recorded 5 minutes per run):
* Index 34: Avoidant 0, Ambivalent 0.1 (secure)
* Index 35: Avoidant 0.9, Ambivalent 0 (strongly avoidant)
* Index 36: Avoidant 0, Ambivalent 0.4 (moderately ambivalent)

Parameter sweep (Recorded 3 minutes per run):
* Index 37 - 44: Avoidant 0.1 to 0.8, Ambivalent 0
* Index 45 - 52: Avoidant 0, Ambivalent 0.1 to 0.8
* Index 53: Avoidant 0, Ambivalent 0

## Not in git

For runs 54 onwards only `meta.json` (parameters, seed, controller, starting needs) is in git. The data arrays (`child_data.npy`, `parent_data.npy`, `controller_*.npy`, 1.0 GB) and the overhead videos (`camera_recording.mp4`, 4.6 GB) stay on the recording machine, to be archived outside git; see the repository `.gitignore`. The per-run measures computed from them are in `analysis/tcds/results` (`gazebo_runs.csv`, `run_checks.csv`, `numbers.json`). `experiments/run_batch.py` uses the files to tell finished runs from unfinished ones, so keep them in place when resuming a batch.
