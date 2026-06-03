import main_analysis as ma
from pathlib import Path
import pandas as pd

df = ma.phase1_ingest_excel(Path('External_User_run_excel/20260423-ExternalUserRun-ISBE-LWFA.xlsx'))
df = ma.phase2_map_files(df, Path('.'), target_date='20260423')

bg_cache = {}
for k, v in df.attrs['backgrounds'].items():
    bg_cache[k] = ma.safe_imread(v)

print("Backgrounds loaded:")
for k, v in df.attrs['backgrounds'].items():
    print(k, v)

bad_shots = []
for shot in df.index:
    si_path = df.at[shot, 'Path_SideImaging']
    if pd.isna(si_path):
        continue
    
    date_val = df.at[shot, 'Giorno']
    date_str = str(date_val) if pd.notna(date_val) else None
    bg = bg_cache.get(('SideImaging', date_str))
    img = ma.safe_imread(si_path)
    
    if img is not None:
        if bg is None or bg.shape != img.shape:
            bad_shots.append((shot, date_str, getattr(bg, 'shape', None), img.shape))

print('Bad shots count:', len(bad_shots))
if len(bad_shots) > 0:
    for b in bad_shots[:5]:
        print(b)
