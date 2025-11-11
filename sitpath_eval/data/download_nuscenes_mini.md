# nuScenes-mini Export Instructions

The nuScenes data is distributed under a research-friendly license that requires a free account. We cannot mirror it here, so please follow the steps below to generate the CSV files expected by this repository.

## 1. Request access
1. Create an account at https://www.nuscenes.org/.
2. Agree to the terms of use and download the `v1.0-mini` split (metadata + samples + maps).
3. Unpack the archives into a directory such as `/data/nuscenes/v1.0-mini`.

## 2. Install the devkit
```bash
pip install nuscenes-devkit
```
The devkit provides utilities for iterating over samples and aggregating agent trajectories.

## 3. Export trajectories to CSV
Use the following helper to dump agent centers (vehicles + pedestrians) for each split:
```bash
python - <<'PY'
from pathlib import Path

import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

DATAROOT = Path('/data/nuscenes/v1.0-mini')
OUTROOT = Path('data/nuscenes_mini')
SPLITS = {
    'train': 'mini_train',
    'val': 'mini_val',
    'test': 'mini_val',  # re-use validation as a lightweight test set
}

nusc = NuScenes(version='v1.0-mini', dataroot=str(DATAROOT), verbose=True)
split_scenes = create_splits_scenes()
for split_name, official_split in SPLITS.items():
    allowed_scenes = set(split_scenes[official_split])
    split_dir = OUTROOT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in nusc.scene:
        if scene['name'] not in allowed_scenes:
            continue
        sample_token = scene['first_sample_token']
        while sample_token:
            sample = nusc.get('sample', sample_token)
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                category = ann['category_name'].split('.')[0]
                if category not in {'vehicle', 'pedestrian'}:
                    continue
                rows.append({
                    'scene_token': scene['token'],
                    'instance_token': ann['instance_token'],
                    'timestamp': ann['timestamp'],
                    'center_x': ann['translation'][0],
                    'center_y': ann['translation'][1],
                })
            sample_token = sample['next']
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(split_dir / 'scene_aggregate.csv', index=False)
PY
```
This produces CSVs with the columns `instance_token,timestamp,center_x,center_y` that the loader expects. Feel free to split scenes into individual files if you prefer finer-grained bookkeeping.

## 4. Directory layout
```
data/
  nuscenes_mini/
    train/
      scene_aggregate.csv
    val/
      scene_aggregate.csv
    test/
      scene_aggregate.csv
```

Once the CSVs are in place, `python scripts/precompute_tokens.py --dataset nuscenes_mini` will automatically pick them up.
