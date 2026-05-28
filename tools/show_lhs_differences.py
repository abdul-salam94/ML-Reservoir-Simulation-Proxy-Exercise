import csv
from pathlib import Path

csv_path = Path(r'D:\NORNE\cases\lhs_design.csv')

with open(csv_path) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

fault_names = header[1:]
print(f'Total faults sampled: {len(fault_names)}')
print(f'Total cases:          {len(rows)}')
print()

# Pick 5 representative cases and 6 representative faults
sample_cases = [0, 99, 249, 399, 499]   # NORNE_001, 100, 250, 400, 500
sample_faults = ['E_01', 'DE_0', 'BC', 'C_08', 'CD_1', 'C_20']
baselines = {'E_01': 0.01, 'DE_0': 20.0, 'BC': 0.1, 'C_08': 0.01, 'CD_1': 0.1, 'C_20': 0.1}

print('=== Per-case fault multipliers (5 cases x 6 faults) ===')
header_str = ' ' * 14
for f in sample_faults:
    header_str += f.rjust(14)
print(header_str)

print('baseline'.ljust(14), end='')
for f in sample_faults:
    print(f'{baselines[f]:14.5f}', end='')
print()
print('-' * 90)

for case_idx in sample_cases:
    row = rows[case_idx]
    case_id = row[0]
    vals = []
    for fname in sample_faults:
        col_idx = fault_names.index(fname)
        vals.append(float(row[col_idx + 1]))
    label = f'NORNE_{case_id}'
    print(label.ljust(14), end='')
    for v in vals:
        print(f'{v:14.5f}', end='')
    print()

print()
print('=== Per-fault statistics across all 500 cases ===')
print(f'{"fault":12s}{"baseline":>12s}{"min":>14s}{"median":>14s}{"max":>14s}{"max/min":>10s}')
for fname in sample_faults:
    col_idx = fault_names.index(fname)
    vals = sorted(float(r[col_idx + 1]) for r in rows)
    print(f'{fname:12s}{baselines[fname]:12.4f}{vals[0]:14.5f}{vals[len(vals)//2]:14.5f}{vals[-1]:14.5f}{vals[-1]/vals[0]:10.1f}x')
