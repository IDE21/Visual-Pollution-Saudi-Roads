# Labelling guide (draft)

**Status:** draft. The rules marked **[owner decides]** need the project owner's decision before any relabelling starts.

This guide comes from the [label audit](../reports/label_audit/report.md). Most disagreements were about what counts as each class, not about box drawing. A short written guide makes relabelling consistent and repeatable.

## General rules

1. **One box per object.** Draw a tight box around the visible damage or object, not around its surroundings.
2. **Label every instance** in the image, including small and distant ones, if a person can recognise it at full resolution.
3. **Separate objects get separate boxes**, even when they touch. A row of barriers is several boxes, one per block. **[owner decides]**: the released labels sometimes use one box for a whole row.
4. **Partly hidden objects:** label them if at least about a third is visible.
5. **When unsure, skip the image and flag it.** Do not guess.

## barriers

**Is:** temporary or permanent objects that block or narrow the road or sidewalk:
- concrete jersey barriers;
- plastic or water-filled barriers;
- concrete blocks;
- sandbags placed as a barrier;
- construction fencing.

**Is not:**
- shop signs;
- road signs;
- building columns, even red and white painted ones;
- plastic bags;
- trees and planters.

**[owner decides]** Concrete ball bollards and fixed posts: label them as barriers or not?

## sidewalks (dilapidated sidewalk)

**Is:** damage to the sidewalk or curb:
- broken or missing curb stones;
- broken or missing tiles;
- piles of loose tiles or bricks on the sidewalk;
- collapsed sidewalk edges.

**Is not:**
- an intact sidewalk;
- an intact curb;
- sand or dirt lying on an intact sidewalk.

## pothole

**Is:** a hole or depression in the road surface:
- including water-filled potholes;
- including dug-out or excavated patches in the road.

**Is not:**
- shadows;
- dark stains or wet patches;
- intact road patches (repaired asphalt with no depression);
- sand piles;
- cracks without a depression.

## Boundary cases

| Case | Rule |
|---|---|
| Drain grate or manhole cover, intact, in the road | Not labelled |
| Manhole cover in the road with broken asphalt around it | pothole, with the box around the broken area |
| Open or broken manhole **on the sidewalk** | **[owner decides]** sidewalks or pothole. The released labels use both. |
| Excavation in the road with a grate inside | pothole |
| Debris or brick pile on the sidewalk | sidewalks |
| Long crack in the road, no depression | Not labelled |
| Broken asphalt at the road edge next to the curb | pothole if there is a depression; otherwise not labelled |

## Process

- **Two passes on a sample.** The owner and a second reviewer (or Claude) label the same 50 images, and we measure agreement. Below about 80%, refine this guide before going further.
- **Never relabel the clean test set C itself.** If we relabel test images, they go into a **separate copy**, C-relabelled. Results are always reported on both the original C and the copy.
