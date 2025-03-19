import psutil
import time
import matplotlib.pyplot as plt

cpu_usage_list = []
time_list = []

start_time = time.time()

try:
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        elapsed_time = time.time() - start_time

        cpu_usage_list.append(cpu_usage)
        time_list.append(elapsed_time)

        print(f"CPU Usage: {cpu_usage:.2f}%")

except KeyboardInterrupt:
    print("\nKeyboard Interrupt detected. Saving the graph...")

    # Plot the CPU usage graph
    plt.figure(figsize=(10, 5))
    plt.plot(time_list, cpu_usage_list, label="CPU Usage (%)", color="blue")
    plt.xlabel("Time (seconds)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.legend()
    plt.grid(True)

    # Save the graph
    plt.savefig("cpu_usage_plot.png")
    print("CPU usage graph saved as 'cpu_usage_plot.png'.")

    # Show the graph
    plt.show()