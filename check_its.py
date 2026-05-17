import json
import glob
import os

print("\n=== ITS Results (corrected) ===\n")
print(f"{'Metric':<28} {'Level coef':<12} {'Level p':<14} {'Slope coef':<12} {'Slope p':<14}")
print("-" * 85)

n_level_sig = 0
n_slope_sig = 0

for f in sorted(glob.glob('outputs/tables/its_*_2022-02-24.json')):
    metric = os.path.basename(f).replace('its_', '').replace('_2022-02-24.json', '')
    d = json.load(open(f))
    lc = d.get('level_change', {})
    sc = d.get('slope_change', {})
    l_coef = lc.get('coef')
    l_p = lc.get('p_value')
    s_coef = sc.get('coef')
    s_p = sc.get('p_value')
    
    l_coef_s = f"{l_coef:>10.3f}" if isinstance(l_coef, (int, float)) else "N/A"
    l_p_s = f"{l_p:.3e}" if isinstance(l_p, (int, float)) else "N/A"
    s_coef_s = f"{s_coef:>10.3f}" if isinstance(s_coef, (int, float)) else "N/A"
    s_p_s = f"{s_p:.3e}" if isinstance(s_p, (int, float)) else "N/A"
    
    sig_l = " *" if isinstance(l_p, (int, float)) and l_p < 0.05 else "  "
    sig_s = " *" if isinstance(s_p, (int, float)) and s_p < 0.05 else "  "
    
    if isinstance(l_p, (int, float)) and l_p < 0.05:
        n_level_sig += 1
    if isinstance(s_p, (int, float)) and s_p < 0.05:
        n_slope_sig += 1
    
    print(f"{metric:<28} {l_coef_s:<12} {l_p_s:<12}{sig_l} {s_coef_s:<12} {s_p_s:<12}{sig_s}")

print(f"\nLevel change significant (p<0.05): {n_level_sig} / 15")
print(f"Slope change significant (p<0.05): {n_slope_sig} / 15")
