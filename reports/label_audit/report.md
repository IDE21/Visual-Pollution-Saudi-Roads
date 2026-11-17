# Label audit on val (plan v2, step 4)

The aim was to measure how many of the released labels are wrong before spending compute on bigger models. Only **val** was audited; the clean test set C was not touched.

## Method

1. **Automatic flagging** ([`src/label_audit.py`](../../src/label_audit.py)). v1 (YOLO11s, which never trained on val) predicted every val image. Rules compared each prediction with the labels:

   | Candidate type | Rule | Count |
   |---|---|---|
   | Possible wrong label | Label with no prediction of its class nearby (IoU ≥ 0.1, conf ≥ 0.1) | 536 |
   | Possible missing label | Prediction with conf ≥ 0.5 and no label of any class at IoU ≥ 0.3 | 447 |
   | Possible bad box | Same class, IoU between 0.1 and 0.5 | 34 |
   | Possible wrong class | A label of another class at IoU ≥ 0.5 | 29 |
   | **Total** | | **1,046** (val has 3,994 labels in 3,053 images) |

2. **Sample.** 100 candidates drawn at random, stratified by type (35/35/15/15), and shuffled ([`sample.csv`](sample.csv)).
3. **Two judges.**
   - Claude judged all 100 crops visually. The verdicts were written to [`claude_verdicts.csv`](claude_verdicts.csv) **before** the owner's review.
   - The project owner judged the first 50 blind, on a review page that showed no verdicts ([`owner_answers.json`](owner_answers.json)).

## Results

### How reliable is the automatic judge?

| | Value |
|---|---|
| Cases Claude could not decide | 25 of 50 (50%) |
| Agreement where Claude did decide | 21 / 25 = **84%** (Cohen's κ = 0.68) |
| Disagreements | 4: cases 12, 35, 41, 47 |

The owner decided all 50. **The automatic judge is not good enough to replace the owner.** It agrees well when it decides, but it abstains on half the cases. Most of those are small or blurry crops, or boundary cases between categories (for example, a broken manhole: pothole or sidewalk?).

### How many labels are wrong? (owner's answers)

| Candidate type | Owner: label is wrong | Estimated in val |
|---|---|---|
| Possible missing label | 13 / 22 (59%) | ~264 of 447 |
| Possible wrong label | 7 / 13 (54%) | ~289 of 536 |
| Possible wrong class | 7 / 9 (78%) | ~23 of 29 |
| Possible bad box | 1 / 6 (17%) | ~6 of 34 |
| **All** | **28 / 50 (56%, 95% CI 42–69%)** | **~580 (CI about 440–720)** |

That is roughly **11–18% of val labels** wrong or missing. This is a **lower bound**: the audit only catches errors where the model disagrees with the label.

## Findings

1. **Label noise is large.** More than half of the model's "errors" on val are actually label errors. That is well above the plan's threshold of one quarter, so cleaning the labels is justified.
2. **Part of our measured error is not model error.** With about 15% of val labels wrong, mAP on these labels underestimates the model. The same is very likely true of the paper's labels, which came from the same active-learning pipeline.
3. **Category definitions are ambiguous.** Manholes, drains, debris and curb damage fall between "pothole" and "sidewalk". A written labelling guide would help any relabelling effort.
4. **Bad boxes are rare.** Most errors are missing or false labels, not loose boxes.

## Limits

- Only 50 cases were judged by a human, and by one person.
- The estimate covers val only. Train is assumed to be similar, since it comes from the same pipeline.
- Candidates were found with one model, so errors that the model shares with the labels are invisible.
