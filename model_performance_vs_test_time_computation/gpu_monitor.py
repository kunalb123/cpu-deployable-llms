import psutil
import time
import matplotlib.pyplot as plt

timestamps = []
gpu_mem_usage = []

print("Tracking unified memory usage (Apple Silicon)... Press Ctrl+C to stop.")
start_time = time.time()

try:
    while True:
        elapsed_time = time.time() - start_time
        unified_mem = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB
        
        timestamps.append(elapsed_time)
        gpu_mem_usage.append(unified_mem)

        print(f"Time: {elapsed_time:.1f}s | Unified Memory Usage: {unified_mem:.2f} MB")

        time.sleep(1)  # Log every 1 seconds

except KeyboardInterrupt:
    print("\nMonitoring stopped.")

# Plot Unified Memory Usage Over Time
plt.figure(figsize=(10, 5))
plt.plot(timestamps, gpu_mem_usage, label="Unified Memory (MB)", color='green')
plt.xlabel("Time (s)")
plt.ylabel("Memory (MB)")
plt.title("Apple Unified Memory Usage Over Time")
plt.legend()
plt.grid()
plt.savefig("apple_gpu_usage.png")
print("GPU usage graph saved as apple_gpu_usage.png")