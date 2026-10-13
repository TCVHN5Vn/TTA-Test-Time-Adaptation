# Raw WSU sensor data

`S1_raw.csv` and `S2_raw.csv` are the two single-occupant apartments used throughout,
from the WSU CASAS smart-home collection (Cook, *Learning setting-generalized activity
models for smart spaces*, IEEE Intelligent Systems 27(1), 2012). The source collection is
public: <https://casas.wsu.edu/datasets/>.

Columns: `time` (seconds), `year_day`, `activity` (15 / 60 / 65 / 70 / 85), and one binary
column per sensor (70 in S1, 75 in S2; 53 shared).

Activity codes map as `{15: toileting, 60: meal_prep, 65: watching_tv, 70: music,
85: cleaning}`, following `LABEL_NAMES` in the original feature-engineering notebook. Note
that these differ from the names used in the earlier manuscript draft; the time-of-day
distributions support the mapping above (see CORRECTIONS.md §5).

Occupancy data is **not** included: it is private (two offices at the Grenoble Institute
of Technology, via Dridi, Amayri & Bouguila). The occupancy scripts will not run without it.
