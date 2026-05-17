import json
import glob
import os

print("\n=== Event Study Results ===\n")
print(f"{'Metric':<30} {'Status':<10} {'CAR':<15}")
print("-" * 60)

for f in sorted(glob.glob('outputs/tables/event_study_*_2022-02-24.json')):
    metric = os.path.basename(f).replace('event_study_', '').replace('_2022-02-24.json', '')
    d = json.load(open(f))
    status = d.get('status', '?')
    car = d.get('cumulative_abnormal', 'N/A')
    if isinstance(car, float):
        car = f"{car:.4f}"
    print(f"{metric:<30} {status:<10} {car}")

print("\n=== ITS Results (p-values) ===\n")
print(f"{'Metric':<30} {'Level p-val':<15} {'Slope p-val':<15}")
print("-" * 60)

for f in sorted(glob.glob('outputs/tables/its_*_2022-02-24.json')):
    metric = os.path.basename(f).replace('its_', '').replace('_2022-02-24.json', '')
    d = json.load(open(f))
    lc = d.get('level_change_coef', {})
    sc = d.get('slope_change_coef', {})
    lp = lc.get('p_value', 'N/A') if isinstance(lc, dict) else 'N/A'
    sp = sc.get('p_value', 'N/A') if isinstance(sc, dict) else 'N/A'
    if isinstance(lp, float):
        lp = f"{lp:.4f}"
    if isinstance(sp, float):
        sp = f"{sp:.4f}"
    sig_l = "*" if isinstance(lc.get('p_value'), float) and lc['p_value'] < 0.05 else " "
    sig_s = "*" if isinstance(sc.get('p_value'), float) and sc['p_value'] < 0.05 else " "
    print(f"{metric:<30} {str(lp):<13}{sig_l}  {str(sp):<13}{sig_s}")

print("\n* indicates p < 0.05")