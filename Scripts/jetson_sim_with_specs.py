import os
import time
import gc
import threading
import pandas as pd
import torch
import psutil
from ultralytics import YOLO

# === CONFIGURATION ===
BASE_DIR    = "/workspace/" # Update to your actual base directory containing fold subdirectories
DATASET_DIR = "/workspace/" # Update to your actual dataset directory containing fold YAML files

OUTPUT_CSV       = "test_eval_metricsv26_jetson_gpu.csv"
TARGET_ALGORITHM = "yolov26"

# === Jetson Nano GPU Simulation Settings ===
torch.set_num_threads(4)

# Limit GPU memory to 2GB (simulating Jetson Nano's shared memory)
torch.cuda.set_per_process_memory_fraction(2.0 / 12.0, device=0)

print("="*60)
print("  Jetson Nano GPU Simulation Environment")
print("="*60)
print(f"  CPU threads  : {torch.get_num_threads()} (Jetson Nano: 4x Cortex-A57)")
print(f"  Memory limit : 4GB (enforced via Docker)")
print(f"  GPU memory   : 2GB fraction (Jetson Nano shared memory)")
print(f"  GPU clock    : ~930MHz (target: 921MHz, <1% difference)")
print(f"  Precision    : FP16 (Jetson Nano default)")
print(f"  Device       : CUDA GPU")
print("="*60 + "\n")

# ================================================================
# Resource Monitor — runs in background thread during inference
# ================================================================
class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval             = interval
        self.cpu_percent_readings = []
        self.ram_percent_readings = []
        self.ram_mb_readings      = []
        self.cpu_temp_readings    = []
        self.gpu_mem_mb_readings  = []
        self.running              = False
        self.thread               = None

    def start(self):
        self.running = True
        psutil.cpu_percent(interval=None)  # prime first call
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor(self):
        while self.running:
            # CPU utilization %
            self.cpu_percent_readings.append(
                psutil.cpu_percent(interval=None)
            )

            # RAM usage
            mem = psutil.virtual_memory()
            self.ram_percent_readings.append(mem.percent)
            self.ram_mb_readings.append(mem.used / 1024 / 1024)

            # GPU memory usage
            try:
                gpu_mem = torch.cuda.memory_allocated(0) / 1024 / 1024
                self.gpu_mem_mb_readings.append(gpu_mem)
            except Exception:
                pass

            # CPU temperature
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name in ['coretemp', 'cpu_thermal', 'k10temp', 'acpitz']:
                        if name in temps:
                            readings = [t.current for t in temps[name]]
                            self.cpu_temp_readings.append(
                                sum(readings) / len(readings)
                            )
                            break
            except Exception:
                pass

            time.sleep(self.interval)

    def get_results(self):
        def safe_mean(lst):
            return round(sum(lst) / len(lst), 2) if lst else None
        def safe_max(lst):
            return round(max(lst), 2) if lst else None

        return {
            "cpu_util_mean_pct"  : safe_mean(self.cpu_percent_readings),
            "cpu_util_max_pct"   : safe_max(self.cpu_percent_readings),
            "ram_usage_mean_pct" : safe_mean(self.ram_percent_readings),
            "ram_usage_max_pct"  : safe_max(self.ram_percent_readings),
            "ram_usage_mean_mb"  : safe_mean(self.ram_mb_readings),
            "ram_usage_max_mb"   : safe_max(self.ram_mb_readings),
            "gpu_mem_mean_mb"    : safe_mean(self.gpu_mem_mb_readings),
            "gpu_mem_max_mb"     : safe_max(self.gpu_mem_mb_readings),
            "cpu_temp_mean_c"    : safe_mean(self.cpu_temp_readings),
            "cpu_temp_max_c"     : safe_max(self.cpu_temp_readings),
        }


# ================================================================
# CPU Cycles helper
# ================================================================
def get_cpu_times():
    process = psutil.Process(os.getpid())
    t = process.cpu_times()
    return t.user + t.system


# ================================================================
# Collect matching fold directories
# ================================================================
model_folders = sorted([
    f for f in os.listdir(BASE_DIR)
    if f.startswith(TARGET_ALGORITHM) and "fold" in f
])

print(f"Found {len(model_folders)} folds: {model_folders}\n")

results = []

# ================================================================
# Process each fold
# ================================================================
for model_folder in model_folders:
    print(f"\n🚀 Evaluating: {model_folder}")
    fold_num       = model_folder.split("fold")[-1]
    model_path     = os.path.join(BASE_DIR, model_folder, "weights", "best.pt")
    fold_yaml_path = os.path.join(DATASET_DIR, f"fold{fold_num}.yaml")

    if not os.path.exists(fold_yaml_path):
        print(f"❌ Missing YAML: {fold_yaml_path}")
        continue

    if not os.path.exists(model_path):
        print(f"❌ Missing model: {model_path}")
        continue

    # Load model fresh for each fold
    model = YOLO(model_path)

    # Start resource monitor
    monitor = ResourceMonitor(interval=0.5)
    monitor.start()

    # Record CPU time before
    cpu_before = get_cpu_times()

    # Timing
    start_wall = time.perf_counter()
    start_cpu  = time.process_time()

    # Run evaluation — GPU + FP16 (Jetson Nano style)
    test_metrics = model.val(
        data=fold_yaml_path,
        split='test',
        verbose=False,
        device=0,         # ← GPU
        workers=0,
        half=True         # ← FP16 (Jetson Nano default)
    )

    # Stop timing
    end_wall = time.perf_counter()
    end_cpu  = time.process_time()

    # Stop monitor
    monitor.stop()
    resources = monitor.get_results()

    # CPU cycles
    cpu_after      = get_cpu_times()
    cpu_cycles_sec = round(cpu_after - cpu_before, 3)

    # Calculate timing metrics
    wall_time_sec   = round(end_wall - start_wall, 3)
    cpu_time_sec    = round(end_cpu  - start_cpu,  3)
    num_images      = 445  # ← update to your actual test set size
    wall_per_img_ms = round((wall_time_sec / num_images) * 1000, 2)
    cpu_per_img_ms  = round((cpu_time_sec  / num_images) * 1000, 2)
    fps             = round(num_images / wall_time_sec, 2)

    # Collect all results
    result = {
        # Identity
        "algorithm"              : TARGET_ALGORITHM,
        "fold"                   : fold_num,
        "simulated_device"       : "Jetson Nano GPU",

        # Detection metrics
        "precision"              : round(test_metrics.box.mp, 3),
        "recall"                 : round(test_metrics.box.mr, 3),
        "f1"                     : round(sum(test_metrics.box.f1)/len(test_metrics.box.f1), 3) if test_metrics.box.f1 else 0.0,
        "map50"                  : round(test_metrics.box.map50, 3),
        "map50_95"               : round(test_metrics.box.map, 3),

        # Timing metrics
        "wall_time_sec"          : wall_time_sec,
        "cpu_time_sec"           : cpu_time_sec,
        "wall_ms_per_image"      : wall_per_img_ms,
        "cpu_ms_per_image"       : cpu_per_img_ms,
        "fps"                    : fps,
        "num_images"             : num_images,

        # CPU cycles
        "cpu_cycles_sec"         : cpu_cycles_sec,
        "cpu_cycles_per_image"   : round(cpu_cycles_sec / num_images, 4),

        # CPU utilization
        "cpu_util_mean_pct"      : resources["cpu_util_mean_pct"],
        "cpu_util_max_pct"       : resources["cpu_util_max_pct"],

        # RAM usage
        "ram_usage_mean_pct"     : resources["ram_usage_mean_pct"],
        "ram_usage_max_pct"      : resources["ram_usage_max_pct"],
        "ram_usage_mean_mb"      : resources["ram_usage_mean_mb"],
        "ram_usage_max_mb"       : resources["ram_usage_max_mb"],

        # GPU memory
        "gpu_mem_mean_mb"        : resources["gpu_mem_mean_mb"],
        "gpu_mem_max_mb"         : resources["gpu_mem_max_mb"],

        # Temperature
        "cpu_temp_mean_c"        : resources["cpu_temp_mean_c"],
        "cpu_temp_max_c"         : resources["cpu_temp_max_c"],

        # Power
        "power_utilization"      : "N/A (not measurable in Docker)",

        # Settings
        "precision_mode"         : "FP16",
        "device"                 : "CUDA GPU",
        "cpu_threads"            : torch.get_num_threads(),
    }

    results.append(result)

    # Print fold summary
    print(f"\n  📊 Fold {fold_num} Results:")
    print(f"     mAP50              : {result['map50']}")
    print(f"     mAP50-95           : {result['map50_95']}")
    print(f"     Precision          : {result['precision']}")
    print(f"     Recall             : {result['recall']}")
    print(f"     F1                 : {result['f1']}")
    print(f"     Wall time          : {wall_time_sec}s / {wall_per_img_ms}ms per image")
    print(f"     FPS                : {fps}")
    print(f"     CPU cycles (sec)   : {cpu_cycles_sec}s total / {result['cpu_cycles_per_image']}s per image")
    print(f"     CPU utilization    : mean {resources['cpu_util_mean_pct']}% / max {resources['cpu_util_max_pct']}%")
    print(f"     RAM usage          : mean {resources['ram_usage_mean_pct']}% / max {resources['ram_usage_max_pct']}%")
    print(f"     RAM usage (MB)     : mean {resources['ram_usage_mean_mb']}MB / max {resources['ram_usage_max_mb']}MB")
    print(f"     GPU memory         : mean {resources['gpu_mem_mean_mb']}MB / max {resources['gpu_mem_max_mb']}MB")
    print(f"     CPU temperature    : {resources['cpu_temp_mean_c']}°C mean / {resources['cpu_temp_max_c']}°C max")

    # Memory cleanup between folds
    print(f"  Cleaning up memory...")
    del model
    del test_metrics
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Memory freed ✅")

# ================================================================
# Save results and print final summary
# ================================================================
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*60)
print("  FINAL SUMMARY (Jetson Nano GPU Simulation)")
print("="*60)
print(f"  Algorithm          : {TARGET_ALGORITHM}")
print(f"  Folds              : {len(results)}")
print(f"  Mean mAP50         : {df['map50'].mean():.3f} ± {df['map50'].std():.3f}")
print(f"  Mean mAP50-95      : {df['map50_95'].mean():.3f} ± {df['map50_95'].std():.3f}")
print(f"  Mean F1            : {df['f1'].mean():.3f} ± {df['f1'].std():.3f}")
print(f"  Mean FPS           : {df['fps'].mean():.2f}")
print(f"  Mean ms/image      : {df['wall_ms_per_image'].mean():.2f} ms")
print(f"  Mean CPU util      : {df['cpu_util_mean_pct'].mean():.1f}%")
print(f"  Mean RAM usage     : {df['ram_usage_mean_pct'].mean():.1f}%")
print(f"  Mean GPU mem       : {df['gpu_mem_mean_mb'].mean():.1f} MB")
if df['cpu_temp_mean_c'].notna().any():
    print(f"  Mean CPU temp      : {df['cpu_temp_mean_c'].mean():.1f}°C")
else:
    print(f"  CPU temperature    : Not available in Docker")
print("="*60)
print(f"\n✅ Results saved to {OUTPUT_CSV}")
print(df[['fold','map50','map50_95','f1','fps','wall_ms_per_image',
          'cpu_util_mean_pct','ram_usage_mean_pct','gpu_mem_mean_mb']])