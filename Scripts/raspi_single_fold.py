import os
import sys
import time
import gc
import threading
import pandas as pd
import torch
import psutil
from ultralytics import YOLO

# === CONFIGURATION ===
BASE_DIR    = "/workspace" # update to your actual base directory
DATASET_DIR = "/workspace" # update to your actual dataset directory containing fold YAML files
TARGET_ALGORITHM = "yolov26"

# Get fold number from command line argument
fold_num = sys.argv[1]
print(f"\n🚀 Processing fold {fold_num}")

torch.set_num_threads(4)

model_folder   = f"{TARGET_ALGORITHM}_train_fold{fold_num}"
model_path     = os.path.join(BASE_DIR, model_folder, "weights", "best.pt")
fold_yaml_path = os.path.join(DATASET_DIR, f"fold{fold_num}.yaml")

if not os.path.exists(fold_yaml_path):
    print(f"❌ Missing YAML: {fold_yaml_path}")
    sys.exit(1)

if not os.path.exists(model_path):
    print(f"❌ Missing model: {model_path}")
    sys.exit(1)

# ================================================================
# Resource Monitor — runs in background thread during inference
# ================================================================
class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.cpu_percent_readings  = []
        self.ram_percent_readings  = []
        self.ram_mb_readings       = []
        self.cpu_temp_readings     = []
        self.running = False
        self.thread  = None

    def start(self):
        self.running = True
        # Prime cpu_percent (first call always returns 0.0)
        psutil.cpu_percent(interval=None)
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

            # CPU temperature (may not work in all Docker setups)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Try common sensor names
                    for name in ['coretemp', 'cpu_thermal', 'k10temp', 'acpitz']:
                        if name in temps:
                            readings = [t.current for t in temps[name]]
                            self.cpu_temp_readings.append(
                                sum(readings) / len(readings)
                            )
                            break
            except Exception:
                pass  # temperature not available in this container

            time.sleep(self.interval)

    def get_results(self):
        def safe_mean(lst): 
            return round(sum(lst)/len(lst), 2) if lst else None
        def safe_max(lst):  
            return round(max(lst), 2) if lst else None

        return {
            "cpu_util_mean_pct"  : safe_mean(self.cpu_percent_readings),
            "cpu_util_max_pct"   : safe_max(self.cpu_percent_readings),
            "ram_usage_mean_pct" : safe_mean(self.ram_percent_readings),
            "ram_usage_max_pct"  : safe_max(self.ram_percent_readings),
            "ram_usage_mean_mb"  : safe_mean(self.ram_mb_readings),
            "ram_usage_max_mb"   : safe_max(self.ram_mb_readings),
            "cpu_temp_mean_c"    : safe_mean(self.cpu_temp_readings),
            "cpu_temp_max_c"     : safe_max(self.cpu_temp_readings),
        }


# ================================================================
# CPU Cycles measurement via process CPU times
# ================================================================
def get_cpu_times():
    process = psutil.Process(os.getpid())
    t = process.cpu_times()
    return t.user + t.system  # total CPU seconds used by this process


# ================================================================
# Main Evaluation
# ================================================================

# Load model
model = YOLO(model_path)

# Start resource monitor
monitor = ResourceMonitor(interval=0.5)
monitor.start()

# Record CPU time before
cpu_before = get_cpu_times()

# Wall clock timing
start_wall = time.perf_counter()
start_cpu  = time.process_time()

# Run evaluation
test_metrics = model.val(
    data=fold_yaml_path,
    split='test',
    verbose=False,
    device='cpu',
    workers=0,
    half=False
)

# Stop timing
end_wall = time.perf_counter()
end_cpu  = time.process_time()

# Stop monitor
monitor.stop()
resources = monitor.get_results()

# CPU cycles approximation
cpu_after       = get_cpu_times()
cpu_cycles_sec  = round(cpu_after - cpu_before, 3)  # total CPU seconds
                                                      # (proxy for CPU cycles)

# Timing calculations
wall_time_sec   = round(end_wall - start_wall, 3)
cpu_time_sec    = round(end_cpu  - start_cpu,  3)
num_images      = 445  # update to your actual test set size
wall_per_img_ms = round((wall_time_sec / num_images) * 1000, 2)
cpu_per_img_ms  = round((cpu_time_sec  / num_images) * 1000, 2)
fps             = round(num_images / wall_time_sec, 2)

# ================================================================
# Collect all results
# ================================================================
result = {
    # Identity
    "algorithm"              : TARGET_ALGORITHM,
    "fold"                   : fold_num,
    "simulated_device"       : "Raspberry Pi 5",

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

    # CPU cycles (proxy: total CPU seconds consumed)
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

    # Temperature (None if not available in container)
    "cpu_temp_mean_c"        : resources["cpu_temp_mean_c"],
    "cpu_temp_max_c"         : resources["cpu_temp_max_c"],

    # Power (not measurable in Docker — noted as N/A)
    "power_utilization"      : "N/A (not measurable in Docker)",

    # Settings
    "precision_mode"         : "FP32",
    "device"                 : "CPU only",
    "cpu_threads"            : torch.get_num_threads(),
}

# Save individual fold result
output_csv = f"fold{fold_num}_results.csv"
pd.DataFrame([result]).to_csv(output_csv, index=False)

# Print summary
print(f"\n  📊 Fold {fold_num} Results:")
print(f"     mAP50              : {result['map50']}")
print(f"     mAP50-95           : {result['map50_95']}")
print(f"     F1                 : {result['f1']}")
print(f"     FPS                : {fps}")
print(f"     ms/image           : {wall_per_img_ms} ms")
print(f"     CPU cycles (sec)   : {cpu_cycles_sec}s total / {result['cpu_cycles_per_image']}s per image")
print(f"     CPU utilization    : mean {resources['cpu_util_mean_pct']}% / max {resources['cpu_util_max_pct']}%")
print(f"     RAM usage          : mean {resources['ram_usage_mean_pct']}% / max {resources['ram_usage_max_pct']}%")
print(f"     RAM usage (MB)     : mean {resources['ram_usage_mean_mb']}MB / max {resources['ram_usage_max_mb']}MB")
print(f"     CPU temperature    : {resources['cpu_temp_mean_c']}°C mean / {resources['cpu_temp_max_c']}°C max")
print(f"\n✅ Saved to {output_csv}")

# Explicit cleanup
del model
del test_metrics
gc.collect()