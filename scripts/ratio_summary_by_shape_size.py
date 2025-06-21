import pandas as pd
from pathlib import Path

DATA = Path('fifty_one/measurements/data/length_analysis_new_split_shai_exuviae_with_yolo.csv')
df = pd.read_csv(DATA)

df['ratio'] = df['real_length_rel_diff'] / df['pixel_rel_diff']

df['pond_id'] = df['image_name'].str.extract(r'(GX\d{6})')

df['pond_shape'] = df['pond_id'].apply(lambda x: 'circle' if x == 'GX010191' else 'square')

rows = []
for shape in ['circle', 'square']:
    for size in ['big', 'small']:
        sub = df[(df['pond_shape'] == shape) & (df['lobster_size'] == size)]
        if len(sub):
            rows.append({
                'pond_shape': shape,
                'lobster_size': size,
                'count': len(sub),
                'mean_ratio': sub['ratio'].mean(),
                'std_ratio': sub['ratio'].std(),
                'median_ratio': sub['ratio'].median()
            })
summary = pd.DataFrame(rows)
print(summary.to_string(index=False, float_format=lambda x: f'{x:.2f}')) 