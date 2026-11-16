# Random split for Track A (randA)

Built by `src/make_random_split.py` (seed 42); protocol in `reports/leak_protocol.md`.

- Images in the official train+val folders: 31,795
- Held out because they share an incident or bytes with the clean test set C: 3,182
- Random pool (duplicates kept): 28,613, split 70/10/20 at the image level

| Set | Images | barriers | sidewalks | pothole | byte-identical to a train image | incident also in train |
|---|---|---|---|---|---|---|
| train | 20,029 | 5,307 | 4,689 | 16,390 | 0 | 0 |
| val | 2,861 | 707 | 673 | 2,343 | 151 | 429 |
| test_R | 5,723 | 1,556 | 1,307 | 4,644 | 315 | 929 |
| test_R_dedup | 5,408 | 1,469 | 1,207 | 4,365 | 0 | 618 |

`test_R_dedup` is `test_R` without images byte-identical to a train image.
