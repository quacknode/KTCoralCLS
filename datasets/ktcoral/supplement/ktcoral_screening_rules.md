# KTCoral inclusion and exclusion rules

This note documents how the cleaned KTCoral patch set was assembled. It accompanies
`ktcoral_patch_provenance.csv` (one row per retained 224×224 patch).

## A. Automatic inclusion gates (before manual screening)

### ReefNet component
- Exact-label annotations for the 36 Koh Tao target genera (`Sorted_ReefNet_label` / genus match).
- At most **10,000** ReefNet patches retained per genus (random selection when more existed).
- Patches are **224×224** crops centred on the annotated point.

### Web supplementation (iNaturalist, CoralNet, GBIF, OpenVerse)
Used only for genera below the fill target after ReefNet selection.
- **Licence:** openly licensed / CC-compatible material only (iNaturalist scrapes excluded all-rights-reserved; OpenVerse/GBIF rows retain the recorded licence string).
- **iNaturalist:** taxon-matched observations; research-grade preferred where used; photo-level downloads keyed by `observation_id` + `photo_id`.
- **CoralNet fill:** genus-matching labelled points from CoralNet sources; 224×224 crops around annotated points.
- **GBIF:** media records for the scientific name; download URL and licence retained in the GBIF fill manifest.
- **OpenVerse:** accepted when the scientific name appeared in title/tags. For **Caulastrea only**, common-name *search queries* (e.g. “trumpet coral”) were allowed because the genus had no exact ReefNet labels; acceptance still required “Caulastrea” or “Caulastraea” in title/tags.

## B. Manual quality screening (web fill only)

Web-sourced images were inspected before inclusion in training patches. **Exclude** images that were not usable underwater colony photographs, including:
- dried / museum specimens;
- trays, rulers, labels, or laboratory setups as the main subject;
- obvious mislabels (wrong taxon relative to the query/source label);
- non-coral or irrelevant scenes;
- unusable quality (extreme blur, heavy obstruction, or crops that do not show the colony).

**Include** underwater (or in-situ aquarium) photographs of living colonies that match the target genus label and licence constraints.

ReefNet-derived patches were not re-screened image-by-image beyond the source dataset’s existing annotations; web fill was the focus of manual QC.

## C. Deduplication and dataset variants
- Exact duplicate files removed via **SHA-256** when building the balanced subset (one keeper per hash).
- **Full set:** 110,833 patches after QC (ReefNet cap ≤10k/genus; no common post-QC cap).
- **Balanced set:** ≤1,000 unique images/genus after dedupe (35,101 images; 27 genera at 1,000; 9 short).

## D. Split assignments in the CSV
- `split_stratified`: original paper 80/10/10 stratified-by-genus random split.
- `split_sourceaware`: parent-image / observation group-held-out split (no shared `split_group` across partitions).

## E. Known reproducibility limits
- Manual QC decisions were **not** logged as a rejected-image list; rejected downloads were deleted from scrape folders rather than archived with reason codes. The retained set is fully listed in the CSV; the rejected set cannot be reconstructed byte-for-byte from this repository alone.
- Many ReefNet parent frames have **no public URL** in this export (survey name + original filename + annotation id are provided instead).
- Cross-platform mirrors (e.g. GBIF/OpenVerse serving iNaturalist photos) are flagged via platform labels such as `GBIF(iNat)` in `provenance_patches.csv` and linked where URLs encode iNaturalist photo ids.
