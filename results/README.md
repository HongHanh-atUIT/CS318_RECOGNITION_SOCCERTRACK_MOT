# results/

Numbers behind the paper tables. All MOT results are on the 6 test videos (1800–1980 s);
pitch results are on the 6 GNSS clips (720–900 s).

| File | Content |
|---|---|
| `mot_test_per_video.csv` | MOTA / MOTP / IDF1 / HOTA / IDs / FP / FN / GT for every detector × extractor × tracker × GTA configuration (56) and both baselines, one row per video. Averages reproduce Tables 1–4. |
| `runtime_per_video.csv` | Seconds per video: online detection + tracking (`online_s`), SOLIDER embedding for GTA (`solider_extract_s`), GTA split/merge (`gta_refine_s`). |
| `pitch_eval_summary.csv` | Pitch-HOTA, E_1:1 (and diagnostic E_frame) per pipeline (Table 5). Produced by `PipelineMOT/eval_pitch.py`. |
| `pitch_eval_per_video.csv` | Same, per clip. |

Read with `pd.read_csv(path, keep_default_na=False)` so that the extractor value `None` is not parsed as missing.
Analysed in `PipelineMOT/analysis.ipynb` and `SoccerMOT_Baseline/analysis.ipynb`.
