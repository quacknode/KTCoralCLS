from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ktcoralcls.genera import GENERA, IMAGE_EXTS, SPLIT_SEED
PATCHES = ROOT / 'datasets' / 'ktcoral' / 'patches'
OUT_PROV = ROOT / 'datasets' / 'ktcoral' / 'provenance_patches.csv'
OUT_SPLITS = ROOT / 'datasets' / 'ktcoral' / 'splits_sourceaware'
HASHES = ROOT / 'datasets' / 'ktcoral' / 'dedupe' / 'file_hashes.csv'
GENUS_TO_LABEL = {g: i for i, g in enumerate(GENERA)}

class UnionFind:

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = (self.find(a), self.find(b))
        if ra != rb:
            self.parent[rb] = ra

def safe(s: str) -> str:
    return ''.join((c if c.isalnum() or c in '-_' else '_' for c in str(s)))

def parse_inat_ids_from_filename(name: str) -> tuple[str | None, str | None]:
    m = re.search('(?:scraped__)?inat_(\\d+)_(\\d+)', name)
    if m:
        return (m.group(1), m.group(2))
    return (None, None)

def parse_coralnet(name: str) -> tuple[str | None, str | None]:
    m = re.search('coralnet__s(\\d+)__i(\\d+)', name)
    if m:
        return (m.group(1), m.group(2))
    return (None, None)

def parse_gbif_key(name: str) -> str | None:
    m = re.search('gbif__gbif_(\\d+)_', name)
    return m.group(1) if m else None

def parse_openverse_id(name: str) -> str | None:
    m = re.search('openverse__ov_([0-9a-fA-F-]+)', name)
    return m.group(1) if m else None

def parse_reefnet_parent(name: str) -> tuple[str | None, str | None]:
    if name.startswith(('coralnet__', 'gbif__', 'openverse__', 'scraped__', 'inat_')):
        return (None, None)
    m = re.match('^(?P<source>.+?)__(?P<stem>.+?)__r\\d+_c\\d+\\.', name)
    if not m:
        return (None, None)
    return (m.group('source'), m.group('stem'))

def inat_photo_from_url(url: str) -> str | None:
    if not isinstance(url, str):
        return None
    m = re.search('/photos/(\\d+)/', url)
    return m.group(1) if m else None

def photographer_from_attribution(attr: str | None) -> str | None:
    if not isinstance(attr, str) or not attr.strip():
        return None
    m = re.match('\\(c\\)\\s*([^,]+)', attr.strip())
    if m:
        return m.group(1).strip()
    return attr.strip()[:80]

def build_provenance() -> pd.DataFrame:
    reef = pd.read_csv(ROOT / 'datasets' / 'ktcoral' / 'manifest.csv')
    reef_parent_to_survey: dict[str, str] = {}
    for r in reef.itertuples(index=False):
        src_safe = safe(r.source)
        stem = Path(str(r.image_name)).stem
        stem_safe = safe(stem)
        reef_parent_to_survey[f'{src_safe}__{stem_safe}'] = str(r.source)
    inat = pd.read_csv(ROOT / 'datasets' / 'ktcoral' / 'scraped_manifest.csv')
    inat_photo_to_obs = {str(int(r.photo_id)): str(int(r.observation_id)) for r in inat.itertuples(index=False) if pd.notna(r.photo_id) and pd.notna(r.observation_id)}
    inat_obs_to_user = {}
    for r in inat.itertuples(index=False):
        if pd.notna(r.observation_id):
            inat_obs_to_user[str(int(r.observation_id))] = photographer_from_attribution(getattr(r, 'attribution', None))
    gbif = pd.read_csv(ROOT / 'datasets' / 'ktcoral' / 'gbif_fill_manifest.csv')
    gbif_key_meta = {str(int(r.gbif_key)): {'url': r.url, 'photo': inat_photo_from_url(str(r.url))} for r in gbif.itertuples(index=False) if pd.notna(r.gbif_key)}
    ov = pd.read_csv(ROOT / 'datasets' / 'ktcoral' / 'openverse_fill_manifest.csv')
    ov_meta = {str(r.openverse_id): {'provider': r.provider, 'url': r.url, 'photo': inat_photo_from_url(str(r.url))} for r in ov.itertuples(index=False)}
    rows: list[dict] = []
    for genus_dir in sorted((p for p in PATCHES.iterdir() if p.is_dir())):
        genus = genus_dir.name
        if genus not in GENUS_TO_LABEL:
            continue
        for path in genus_dir.iterdir():
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            name = path.name
            rel = f'{genus}/{name}'
            platform = 'ReefNet'
            collection = None
            parent_key = None
            extra = {}
            obs, photo = parse_inat_ids_from_filename(name)
            if obs is not None:
                platform = 'iNaturalist'
                parent_key = f'inat_obs:{obs}'
                collection = f"inat_user:{inat_obs_to_user.get(obs) or 'unknown'}"
                extra = {'observation_id': obs, 'photo_id': photo}
            elif name.startswith('coralnet__'):
                platform = 'CoralNet'
                sid, iid = parse_coralnet(name)
                parent_key = f'coralnet_img:{iid}' if iid else f'coralnet_file:{name}'
                collection = f'coralnet_source:{sid}' if sid else parent_key
                extra = {'coralnet_source_id': sid, 'coralnet_image_id': iid}
            elif name.startswith('gbif__'):
                platform = 'GBIF'
                key = parse_gbif_key(name)
                parent_key = f'gbif:{key}' if key else f'gbif_file:{name}'
                meta = gbif_key_meta.get(key or '', {})
                photo = meta.get('photo')
                if photo and photo in inat_photo_to_obs:
                    parent_key = f'inat_obs:{inat_photo_to_obs[photo]}'
                    collection = f"inat_user:{inat_obs_to_user.get(inat_photo_to_obs[photo]) or 'unknown'}"
                    platform = 'GBIF(iNat)'
                else:
                    collection = 'gbif'
                extra = {'gbif_key': key, 'inat_photo_id': photo}
            elif name.startswith('openverse__'):
                platform = 'OpenVerse'
                oid = parse_openverse_id(name)
                parent_key = f'openverse:{oid}' if oid else f'openverse_file:{name}'
                meta = ov_meta.get(oid or '', {})
                collection = f"openverse_provider:{meta.get('provider') or 'unknown'}"
                photo = meta.get('photo')
                if photo and photo in inat_photo_to_obs:
                    parent_key = f'inat_obs:{inat_photo_to_obs[photo]}'
                    collection = f"inat_user:{inat_obs_to_user.get(inat_photo_to_obs[photo]) or 'unknown'}"
                    platform = 'OpenVerse(iNat)'
                extra = {'openverse_id': oid, 'inat_photo_id': photo}
            else:
                src, stem = parse_reefnet_parent(name)
                if src and stem:
                    platform = 'ReefNet'
                    parent_key = f'reefnet_img:{src}__{stem}'
                    survey = reef_parent_to_survey.get(f'{src}__{stem}', src.replace('_', ' '))
                    collection = f'reefnet_survey:{survey}'
                    extra = {'reefnet_source_safe': src, 'image_stem': stem, 'survey': survey}
                else:
                    platform = 'Unknown'
                    parent_key = f'file:{rel}'
                    collection = parent_key
            rows.append({'path': rel, 'genus': genus, 'label': GENUS_TO_LABEL[genus], 'platform': platform, 'collection': collection, 'parent_key': parent_key, **extra})
    df = pd.DataFrame(rows)
    uf = UnionFind()
    for pk in df['parent_key'].unique():
        uf.add(str(pk))
    if HASHES.exists():
        hashes = pd.read_csv(HASHES)
        path_to_parent = dict(zip(df['path'], df['parent_key']))
        for sha, grp in hashes.groupby('sha256'):
            parents = []
            for p in grp['path'].astype(str):
                pp = p.replace('\\', '/')
                if pp in path_to_parent:
                    parents.append(path_to_parent[pp])
            if len(parents) >= 2:
                root_p = parents[0]
                for other in parents[1:]:
                    uf.union(str(root_p), str(other))
    df['split_group'] = df['parent_key'].map(lambda x: f'grp:{uf.find(str(x))}')
    return df

def assign_splits(df: pd.DataFrame, *, test_size: float=0.1, val_size: float=0.1, seed: int=SPLIT_SEED, max_tries: int=80) -> tuple[pd.DataFrame, dict]:

    def try_seed(s: int) -> tuple[pd.DataFrame, dict] | None:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=s)
        train_val_idx, test_idx = next(gss1.split(df, df['label'], groups=df['split_group']))
        train_val = df.iloc[train_val_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        val_frac = val_size / (1.0 - test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=s)
        train_idx, val_idx = next(gss2.split(train_val, train_val['label'], groups=train_val['split_group']))
        train = train_val.iloc[train_idx].reset_index(drop=True)
        val = train_val.iloc[val_idx].reset_index(drop=True)
        assert set(train['split_group']).isdisjoint(set(val['split_group']))
        assert set(train['split_group']).isdisjoint(set(test['split_group']))
        assert set(val['split_group']).isdisjoint(set(test['split_group']))
        train_genera = set(train['genus'])
        if train_genera != set(GENERA):
            return None
        out = pd.concat([train.assign(split='train'), val.assign(split='val'), test.assign(split='test')], ignore_index=True)
        report = {'seed': s, 'n_train': int(len(train)), 'n_val': int(len(val)), 'n_test': int(len(test)), 'n_groups_train': int(train['split_group'].nunique()), 'n_groups_val': int(val['split_group'].nunique()), 'n_groups_test': int(test['split_group'].nunique()), 'genera_in_train': sorted(train_genera), 'genera_missing_from_test': sorted(set(GENERA) - set(test['genus'])), 'genera_missing_from_val': sorted(set(GENERA) - set(val['genus'])), 'test_genus_counts': test['genus'].value_counts().to_dict(), 'platform_by_split': out.groupby(['split', 'platform']).size().unstack(fill_value=0).to_dict()}
        return (out, report)
    for s in [seed] + [seed + i for i in range(1, max_tries)]:
        result = try_seed(s)
        if result is not None:
            return result
    raise RuntimeError('Could not find a group split with all 36 genera in training')

def leave_one_platform_feasibility(df: pd.DataFrame) -> dict:
    out = {}
    platforms = sorted(df['platform'].str.replace('\\(iNat\\)', '', regex=True).unique())
    for plat in sorted(df['platform'].unique()):
        train = df[df['platform'] != plat]
        missing = sorted(set(GENERA) - set(train['genus']))
        out[plat] = {'n_images': int((df['platform'] == plat).sum()), 'genera_lost_if_held_out': missing, 'n_genera_lost': len(missing)}
    return out

def main() -> None:
    print('Building provenance for', PATCHES)
    df = build_provenance()
    print(f"  patches={len(df):,}  groups={df['split_group'].nunique():,}  platforms={df['platform'].value_counts().to_dict()}")
    OUT_PROV.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ['path', 'genus', 'label', 'platform', 'collection', 'parent_key', 'split_group']
    df[keep_cols].to_csv(OUT_PROV, index=False)
    print(f'  wrote {OUT_PROV}')
    loso = leave_one_platform_feasibility(df)
    print('\nLeave-one-platform-out feasibility (genera lost from train if platform held out):')
    for plat, info in loso.items():
        print(f"  {plat}: n={info['n_images']:,}  lost={info['n_genera_lost']} {info['genera_lost_if_held_out']}")
    print('\nBuilding group-held-out 80/10/10 split...')
    split_df, report = assign_splits(df)
    report['grouping'] = 'parent_image_or_observation: ReefNet parent image; iNat observation; CoralNet image_id; GBIF key (unioned with iNat obs when URL is an iNat photo); OpenVerse id (same); SHA-256 exact duplicates unioned.'
    report['note'] = 'This is NOT leave-one-platform-out. Platform holdout would drop genera that only exist on the held-out platform. See leave_one_platform_out in this report.'
    report['leave_one_platform_out'] = loso
    report['n_patches'] = int(len(df))
    report['n_groups'] = int(df['split_group'].nunique())
    report['platform_counts'] = df['platform'].value_counts().to_dict()
    ct = split_df.groupby(['genus', 'split']).size().unstack(fill_value=0).reindex(GENERA).fillna(0).astype(int)
    report['genus_split_counts'] = ct.to_dict()
    OUT_SPLITS.mkdir(parents=True, exist_ok=True)
    for name in ('train', 'val', 'test'):
        part = split_df[split_df['split'] == name][['path', 'genus', 'label', 'split']].reset_index(drop=True)
        part.to_csv(OUT_SPLITS / f'{name}.csv', index=False)
        print(f"  {name}: {len(part):,} images, {split_df.loc[split_df['split'] == name, 'split_group'].nunique():,} groups")
    split_df.to_csv(OUT_SPLITS / 'all_with_provenance.csv', index=False)
    (OUT_SPLITS / 'split_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'  wrote {OUT_SPLITS}/')
    print(f"  seed used: {report['seed']}")
    print(f"  genera missing from test: {report['genera_missing_from_test']}")
if __name__ == '__main__':
    main()
