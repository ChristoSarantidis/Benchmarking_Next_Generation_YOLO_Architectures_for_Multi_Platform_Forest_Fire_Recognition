import pandas as pd
import glob

csv_files = sorted(glob.glob("fold*_results.csv"))
print(f"Merging {len(csv_files)} fold results...")

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df.to_csv("test_eval_metrics_rpi5_final.csv", index=False)

print("\n" + "="*70)
print("  FINAL SUMMARY (Raspberry Pi 5 Simulation)")
print("="*70)
print(f"  Mean mAP50             : {df['map50'].mean():.3f} ± {df['map50'].std():.3f}")
print(f"  Mean mAP50-95          : {df['map50_95'].mean():.3f} ± {df['map50_95'].std():.3f}")
print(f"  Mean F1                : {df['f1'].mean():.3f} ± {df['f1'].std():.3f}")
print(f"  Mean FPS               : {df['fps'].mean():.2f}")
print(f"  Mean ms/image          : {df['wall_ms_per_image'].mean():.2f} ms")
print(f"  Mean CPU cycles/image  : {df['cpu_cycles_per_image'].mean():.4f} s")
print(f"  Mean CPU utilization   : {df['cpu_util_mean_pct'].mean():.1f}%")
print(f"  Mean RAM usage         : {df['ram_usage_mean_pct'].mean():.1f}%")
print(f"  Mean RAM usage (MB)    : {df['ram_usage_mean_mb'].mean():.1f} MB")
if df['cpu_temp_mean_c'].notna().any():
    print(f"  Mean CPU temperature   : {df['cpu_temp_mean_c'].mean():.1f}°C")
else:
    print(f"  CPU temperature        : Not available in Docker")
print(f"  Power utilization      : Not measurable in Docker")
print("="*70)
print(f"\n✅ Final results saved to test_eval_metrics_rpi5_final.csv")
print(df[['fold','map50','f1','fps','cpu_util_mean_pct','ram_usage_mean_pct','cpu_cycles_per_image']])