import psutil
import time
import matplotlib.pyplot as plt
import os
import gc

def get_python_ram_usage():
    """Returns the total RAM usage (including child processes) of the current Python session."""
    process = psutil.Process(os.getpid())  # Get the main Python process
    mem_usage = process.memory_info().rss  # Main process RAM usage

    # Add memory usage of child processes (e.g., if llama-cpp-python spawns them)
    for child in process.children(recursive=True):
        try:
            mem_usage += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass  # If child process ended, ignore

    return mem_usage / (1024 * 1024)  # Convert to MB

# Initialize lists for tracking RAM usage
timestamps = []
ram_usages = []

start_time = time.time()
print("Monitoring RAM usage...")

try:
    while True:
        gc.collect()  # Force garbage collection before measuring RAM
        ram_usage = get_python_ram_usage()
        elapsed_time = time.time() - start_time  # Get elapsed time in seconds

        print(f"Time: {elapsed_time:.1f}s | RAM Usage: {ram_usage:.2f} MB")
        timestamps.append(elapsed_time)
        ram_usages.append(ram_usage)

        time.sleep(0.5)  # Check RAM usage every 2 seconds

except KeyboardInterrupt:
    print("\nMonitoring stopped by user.")

# Plot and save the graph
if timestamps:
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, ram_usages, marker='o', linestyle='-', color='b', label="RAM Usage (MB)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("RAM Usage (MB)")
    plt.title("Python Process RAM Usage Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("ram_usage.png")  # Save the graph
    print("Graph saved as ram_usage.png")
else:
    print("No data collected.")