from __future__ import annotations
import re
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ktcoralcls.genera import GENERA
OUT_DIR = ROOT / 'datasets' / 'ktcoral' / 'supplement'
OUT_CSV = OUT_DIR / 'ktcoral_patch_provenance.csv'
OUT_RULES = OUT_DIR / 'ktcoral_screening_rules.md'

def safe(s: str) -> str:
    return ''.join((c if c.isalnum() or c in '-_' else '_' for c in str(s)))

def load_splits(name: str) -> pd.Series:
    base = ROOT / 'datasets' / 'ktcoral' / name
    parts = []
    for split in ('train', 'val', 'test'):
        p = base / f'{split}.csv'
        if p.exists():
            df = pd.read_csv(p, usecols=['path', 'split'])
            parts.append(df)
    if not parts:
        return pd.Series(dtype=str)
    all_df = pd.concat(parts, ignore_index=True)
    return all_df.drop_duplicates('path').set_index('path')['split']

def build_source_lookups() -> dict[str, pd.DataFrame]:
    inat = pd.read_csv(ROOT / 'datasets' / 'ktcoral' / 'scraped_manifest.csv')
    inat['key'] = inat['filename'].astype(str).map(lambda f: Path(f).name)
    inat_by_file = inat.drop_duplicates('key').set_index('key')
    cn = pd.read_csv(ROOT / 'datasets' / 'ktcoral' / 'coralnet_fill_manifest.csv')
    cn['key'] = cn['filename'].astype(str).map(lambda f: Path(f).name)
    cn_by_file = cn.drop_duplicates('key').set_index('key')
    gbif = pd.read_csv(ROOT / 'datasets' / 'ktcoral' / 'gbif_fill_manifest.csv')
    gbif['key'] = gbif['filename'].astype(str).map(lambda f: Path(f).name)
    gbif_by_file = gbif.drop_duplicates('key').set_index('key')
    ov = pd.read_csv(ROOT / 'datasets' / 'ktcoral' / 'openverse_fill_manifest.csv')
    ov['key'] = ov['filename'].astype(str).map(lambda f: Path(f).name)
    ov_by_file = ov.drop_duplicates('key').set_index('key')
    reef = pd.read_csv(ROOT / 'datasets' / 'ktcoral' / 'manifest.csv')
    reef['stem'] = reef['image_name'].map(lambda x: Path(str(x)).stem)
    reef['join_key'] = reef.apply(lambda r: f"{safe(r['source'])}__{safe(r['stem'])}__r{int(r['row'])}_c{int(r['column'])}", axis=1)
    reef_by_key = reef.drop_duplicates('join_key').set_index('join_key')
    return {'inat': inat_by_file, 'cn': cn_by_file, 'gbif': gbif_by_file, 'ov': ov_by_file, 'reef': reef_by_key}

def enrich_row(path: str, platform: str, lookups: dict) -> dict:
    name = Path(path).name
    out = {'source_record_id': '', 'source_url': '', 'license': '', 'attribution_or_provider': '', 'original_filename': ''}
    if platform.startswith('iNaturalist') or name.startswith('scraped__inat_') or 'inat_' in name:
        key = re.sub('^scraped__', '', name)
        row = lookups['inat'].loc[key] if key in lookups['inat'].index else None
        if row is None:
            m = re.search('inat_(\\d+)_(\\d+)', name)
            if m:
                out['source_record_id'] = f'inat_obs:{m.group(1)};photo:{m.group(2)}'
                out['source_url'] = f'https://www.inaturalist.org/observations/{m.group(1)}'
            return out
        out['source_record_id'] = f"inat_obs:{int(row['observation_id'])};photo:{int(row['photo_id'])}"
        out['source_url'] = str(row.get('source_url') or '')
        out['license'] = str(row.get('license') or '')
        out['attribution_or_provider'] = str(row.get('attribution') or '')
        out['original_filename'] = str(row.get('filename') or '')
        return out
    if name.startswith('coralnet__'):
        row = lookups['cn'].loc[name] if name in lookups['cn'].index else None
        if row is not None:
            iid = int(row['image_id'])
            out['source_record_id'] = f"coralnet_source:{int(row['source_id'])};image:{iid};point:{int(row['point_pk'])}"
            out['source_url'] = f'https://coralnet.ucsd.edu/image/{iid}/view/'
            out['license'] = 'CoralNet source terms (see originating CoralNet source)'
            out['attribution_or_provider'] = f"CoralNet source_id={int(row['source_id'])}"
            out['original_filename'] = str(row.get('image_name') or '')
        return out
    if name.startswith('gbif__'):
        row = lookups['gbif'].loc[name] if name in lookups['gbif'].index else None
        if row is not None:
            out['source_record_id'] = f"gbif:{int(row['gbif_key'])}"
            out['source_url'] = str(row.get('url') or '')
            out['license'] = str(row.get('license') or '')
            out['attribution_or_provider'] = str(row.get('publisher') or row.get('scientific_name') or '')
            out['original_filename'] = str(row.get('filename') or '')
        return out
    if name.startswith('openverse__'):
        row = lookups['ov'].loc[name] if name in lookups['ov'].index else None
        if row is not None:
            oid = str(row['openverse_id'])
            out['source_record_id'] = f'openverse:{oid}'
            out['source_url'] = str(row.get('foreign_landing_url') or row.get('url') or '')
            lic = str(row.get('license') or '')
            ver = str(row.get('license_version') or '')
            out['license'] = f'{lic} {ver}'.strip()
            out['attribution_or_provider'] = str(row.get('provider') or '')
            out['original_filename'] = str(row.get('title') or '')
        return out
    m = re.match('^(?P<source>.+?)__(?P<stem>.+?)__r(?P<row>\\d+)_c(?P<col>\\d+)\\.', name)
    if m:
        key = f"{m.group('source')}__{m.group('stem')}__r{m.group('row')}_c{m.group('col')}"
        row = lookups['reef'].loc[key] if key in lookups['reef'].index else None
        if row is not None:
            out['source_record_id'] = f"reefnet_ann:{int(row['annotation_id'])}"
            out['attribution_or_provider'] = str(row['source'])
            out['original_filename'] = str(row['image_name'])
            out['license'] = 'ReefNet / originating CoralNet survey terms'
            out['source_url'] = ''
        else:
            out['source_record_id'] = f'reefnet_patch:{name}'
            out['license'] = 'ReefNet / originating CoralNet survey terms'
            out['attribution_or_provider'] = m.group('source')
    return out

def write_rules() -> None:
    text = '# KTCoral inclusion and exclusion rules\n\nThis note documents how the cleaned KTCoral patch set was assembled. It accompanies\n`ktcoral_patch_provenance.csv` (one row per retained 224×224 patch).\n\n## A. Automatic inclusion gates (before manual screening)\n\n### ReefNet component\n- Exact-label annotations for the 36 Koh Tao target genera (`Sorted_ReefNet_label` / genus match).\n- At most **10,000** ReefNet patches retained per genus (random selection when more existed).\n- Patches are **224×224** crops centred on the annotated point.\n\n### Web supplementation (iNaturalist, CoralNet, GBIF, OpenVerse)\nUsed only for genera below the fill target after ReefNet selection.\n- **Licence:** openly licensed / CC-compatible material only (iNaturalist scrapes excluded all-rights-reserved; OpenVerse/GBIF rows retain the recorded licence string).\n- **iNaturalist:** taxon-matched observations; research-grade preferred where used; photo-level downloads keyed by `observation_id` + `photo_id`.\n- **CoralNet fill:** genus-matching labelled points from CoralNet sources; 224×224 crops around annotated points.\n- **GBIF:** media records for the scientific name; download URL and licence retained in the GBIF fill manifest.\n- **OpenVerse:** accepted when the scientific name appeared in title/tags. For **Caulastrea only**, common-name *search queries* (e.g. “trumpet coral”) were allowed because the genus had no exact ReefNet labels; acceptance still required “Caulastrea” or “Caulastraea” in title/tags.\n\n## B. Manual quality screening (web fill only)\n\nWeb-sourced images were inspected before inclusion in training patches. **Exclude** images that were not usable underwater colony photographs, including:\n- dried / museum specimens;\n- trays, rulers, labels, or laboratory setups as the main subject;\n- obvious mislabels (wrong taxon relative to the query/source label);\n- non-coral or irrelevant scenes;\n- unusable quality (extreme blur, heavy obstruction, or crops that do not show the colony).\n\n**Include** underwater (or in-situ aquarium) photographs of living colonies that match the target genus label and licence constraints.\n\nReefNet-derived patches were not re-screened image-by-image beyond the source dataset’s existing annotations; web fill was the focus of manual QC.\n\n## C. Deduplication and dataset variants\n- Exact duplicate files removed via **SHA-256** when building the balanced subset (one keeper per hash).\n- **Full set:** 110,833 patches after QC (ReefNet cap ≤10k/genus; no common post-QC cap).\n- **Balanced set:** ≤1,000 unique images/genus after dedupe (35,101 images; 27 genera at 1,000; 9 short).\n\n## D. Split assignments in the CSV\n- `split_stratified`: original paper 80/10/10 stratified-by-genus random split.\n- `split_sourceaware`: parent-image / observation group-held-out split (no shared `split_group` across partitions).\n\n## E. Known reproducibility limits\n- Manual QC decisions were **not** logged as a rejected-image list; rejected downloads were deleted from scrape folders rather than archived with reason codes. The retained set is fully listed in the CSV; the rejected set cannot be reconstructed byte-for-byte from this repository alone.\n- Many ReefNet parent frames have **no public URL** in this export (survey name + original filename + annotation id are provided instead).\n- Cross-platform mirrors (e.g. GBIF/OpenVerse serving iNaturalist photos) are flagged via platform labels such as `GBIF(iNat)` in `provenance_patches.csv` and linked where URLs encode iNaturalist photo ids.\n'
    OUT_RULES.write_text(text, encoding='utf-8')

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Loading provenance + manifests...')
    prov = pd.read_csv(ROOT / 'datasets' / 'ktcoral' / 'provenance_patches.csv')
    growth = pd.read_csv(ROOT / 'datasets' / 'ktcoral' / 'patch_growth_labels.csv', usecols=['path', 'growth', 'growth_id'])
    lookups = build_source_lookups()
    split_s = load_splits('splits')
    split_sa = load_splits('splits_sourceaware')
    split_b = load_splits('splits_balanced')
    print(f'Enriching {len(prov):,} patches...')
    records = []
    for r in prov.itertuples(index=False):
        meta = enrich_row(r.path, r.platform, lookups)
        is_reef = str(r.platform).startswith('ReefNet')
        records.append({'image_id': r.path, 'genus': r.genus, 'platform': r.platform, 'collection': r.collection, 'parent_key': r.parent_key, 'split_group': r.split_group, 'origin': 'ReefNet' if is_reef else 'supplement', 'source_record_id': meta['source_record_id'], 'source_url': meta['source_url'], 'license': meta['license'], 'attribution_or_provider': meta['attribution_or_provider'], 'original_source_filename': meta['original_filename'], 'has_growth_form': False, 'growth_form': '', 'split_stratified': split_s.get(r.path, ''), 'split_sourceaware': split_sa.get(r.path, ''), 'split_balanced': split_b.get(r.path, '')})
    out = pd.DataFrame(records)
    gmap = growth.drop_duplicates('path').set_index('path')
    out['growth_form'] = out['image_id'].map(lambda p: gmap.at[p, 'growth'] if p in gmap.index and pd.notna(gmap.at[p, 'growth']) else '')
    out['has_growth_form'] = out['growth_form'].astype(str).str.len() > 0
    out.to_csv(OUT_CSV, index=False)
    write_rules()
    n = len(out)
    n_url = (out['source_url'].astype(str).str.len() > 0).sum()
    n_lic = (out['license'].astype(str).str.len() > 0).sum()
    print(f'wrote {OUT_CSV} ({n:,} rows)')
    print(f'  with source_url: {n_url:,} ({100 * n_url / n:.1f}%)')
    print(f'  with license:    {n_lic:,} ({100 * n_lic / n:.1f}%)')
    print(f"  supplement:      {(out.origin == 'supplement').sum():,}")
    print(f'  with growth:     {out.has_growth_form.sum():,}')
    print(f'wrote {OUT_RULES}')
if __name__ == '__main__':
    main()
