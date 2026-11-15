# Dataset splits (v1)

Built by `src/make_splits.py` (seed 42) from the official train and val folders of the Saudi Arabia Public Roads Visual Pollution Dataset.

## How they were built

- Images pooled: 31,795
- Exact duplicates removed (identical bytes): 1,271
- Duplicates whose label files differ: 0 (first copy kept)
- Unique images: 30,524 in 27,893 incident groups
- Whole incident groups are assigned to one split, so no incident and no identical image appears in two splits (checked by the script).

## Result

| Split | Images | Incidents | barriers | sidewalks | pothole | objects |
|---|---|---|---|---|---|---|
| train | 24,418 (80.0%) | 22,317 | 6,453 (80.0%) | 5,581 (80.0%) | 19,910 (80.0%) | 31,944 |
| val | 3,053 (10.0%) | 2,790 | 807 (10.0%) | 698 (10.0%) | 2,489 (10.0%) | 3,994 |
| test | 3,053 (10.0%) | 2,786 | 807 (10.0%) | 698 (10.0%) | 2,488 (10.0%) | 3,993 |

Percentages are the share of each class's objects that landed in that split (target: train 80%, val 10%, test 10%).

Near-duplicate photos that are not byte-identical and carry different incident ids can still exist across splits; see `reports/data_audit/` for candidates.
