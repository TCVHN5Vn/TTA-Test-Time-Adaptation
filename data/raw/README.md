# Raw sensor data

`S1_raw.csv` and `S2_raw.csv` are the two single-occupant apartments used throughout,
from E. M. Tapia, S. S. Intille & K. Larson, *Activity recognition in the home using
simple and ubiquitous sensors*, Pervasive Computing 2004 (MIT House_n). Binary proximity
sensors were placed on household objects and furniture in two apartments and recorded
over roughly two weeks each.

Note on attribution: this dataset is referred to as "the WSU dataset" in several related
papers from this group. That is a mislabel. Tapia, Intille and Larson were at MIT;
Washington State University is the home of the separate CASAS collection (D. J. Cook).
The activity vocabulary here — preparing meals, toileting, watching TV, listening to
music, cleaning, bathing, grooming, doing laundry — is Tapia's.

Columns: `time` (seconds), `year_day`, `activity` (15 / 60 / 65 / 70 / 85), and one binary
column per sensor (70 in S1, 75 in S2; 53 shared).

Activity codes map as `{15: toileting, 60: meal_prep, 65: watching_tv, 70: music,
85: cleaning}`. This matches the five known classes listed by Abderrahim et al. (IBPSA
2025) for the same data — "preparing food, toileting, listening to music, watching TV,
and cleaning" — and is supported by the time-of-day distributions.

The dataset contains **two apartments and no more**, which bounds what can be said about
generalisation beyond them.

Occupancy data is **not** included: it is private (two offices at the Grenoble Institute
of Technology, via Dridi, Amayri & Bouguila). The occupancy scripts will not run without it.
