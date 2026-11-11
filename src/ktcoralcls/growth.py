from __future__ import annotations
import pandas as pd
COARSE_FORMS = ('Branching', 'Massive', 'Encrusting', 'Foliose', 'Submassive')
BRANCHING_SUBTYPES = ('Tabulate', 'Digitate', 'Corymbose', 'Caespitose', 'Arborescent')
GROWTH_CLASSES: list[str] = ['Massive', 'Encrusting', 'Foliose', 'Submassive', 'Branching', *BRANCHING_SUBTYPES]
GROWTH_TO_INDEX = {name: i for i, name in enumerate(GROWTH_CLASSES)}
IGNORE_INDEX = -100

def normalize_growth(growthform: str | None, branching_spec: str | None) -> str | None:
    spec = '' if branching_spec is None or pd.isna(branching_spec) else str(branching_spec).strip()
    form = '' if growthform is None or pd.isna(growthform) else str(growthform).strip()
    if spec in BRANCHING_SUBTYPES:
        return spec
    if form in COARSE_FORMS:
        return form
    return None

def growth_to_id(name: str | None) -> int:
    if name is None or (isinstance(name, float) and pd.isna(name)) or name == '':
        return IGNORE_INDEX
    return GROWTH_TO_INDEX[str(name)]
