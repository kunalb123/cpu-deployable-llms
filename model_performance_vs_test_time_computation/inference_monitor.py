import psutil
import json
import time
import matplotlib.pyplot as plt
import gc
import threading
import os
from cpu_model import CPUModel


def load_prompts(file_path):
    """Loads the list of prompt strings from a JSON file."""
    with open(file_path, 'r') as f:
        prompts = json.load(f)
    return prompts

output_file = 'new_prompts.json'
loaded_prompts = load_prompts(output_file)

MODEL_FILES = [
    '/Users/kunalbhandarkar/Downloads/DeepSeek-R1-Distill-Llama-8B-Q2_K_L.gguf',
    '/Users/kunalbhandarkar/Downloads/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf',
    '/Users/kunalbhandarkar/Downloads/DeepSeek-R1-Distill-Qwen-7B-Q2_K.gguf',
    '/Users/kunalbhandarkar/Downloads/DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf',
    # '/Users/kunalbhandarkar/Downloads/deepseek-r1-distill-llama-8b-q2_k.gguf',
    # '/Users/kunalbhandarkar/Downloads/Llama-3.2-1B-Instruct.Q5_K_S.gguf',           # smaller llama model
    # '/Users/kunalbhandarkar/Downloads/Llama-3.2-1B-Instruct.Q8_0.gguf',             # larger llama model
    '/Users/kunalbhandarkar/Downloads/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf',     # smaller supposedly better model
    '/Users/kunalbhandarkar/Downloads/DeepSeek-R1-Distill-Qwen-1.5B-f32.gguf',      # 3 GB model
    '/Users/kunalbhandarkar/Downloads/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf',     # larger size model
    '/Users/kunalbhandarkar/Downloads/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf'
]

def get_ram_usage():
    """Returns the current RAM usage in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)  # Convert bytes to MB


def get_unified_memory_usage():
    """Returns the unified memory usage on Apple Silicon in MB."""
    return psutil.virtual_memory().used / (1024 * 1024)  # Convert bytes to MB


def track_resources(stop_event, timestamps, cpu_usage_list, ram_usage_list, unified_mem_usage_list, start_time):
    """Monitors CPU, RAM, and Unified Memory usage while inference runs."""
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        ram_usage = get_ram_usage()
        unified_mem = get_unified_memory_usage()
        cpu_usage = psutil.cpu_percent(interval=0.1)  # Quick CPU sampling

        timestamps.append(elapsed_time)
        ram_usage_list.append(ram_usage)
        unified_mem_usage_list.append(unified_mem)
        cpu_usage_list.append(cpu_usage)

        print(f"Time: {elapsed_time:.1f}s | CPU: {cpu_usage:.2f}% | RAM: {ram_usage:.2f} MB | Unified Mem: {unified_mem:.2f} MB")


def monitor_inference(model_file):
    model_name = model_file.split('/')[-1]  # Extract filename
    os.makedirs(model_name, exist_ok=True)
    log_filename = f"{model_name}/log.txt"
    if os.path.exists(log_filename): os.remove(log_filename)


    def log_and_print(message):
        """Prints message to console and writes it to the log file."""
        print(message)
        with open(log_filename, "a") as log_file:
            log_file.write(message + "\n")

    timestamps = []
    cpu_usage_list = []
    ram_usage_list = []
    unified_mem_usage_list = []

    log_and_print(f"\n### Running inference on: {model_name} ###\n")
    log_and_print(f"Initial RAM Usage: {get_ram_usage():.2f} MB")
    log_and_print(f"Initial Unified Memory Usage: {get_unified_memory_usage():.2f} MB")

    # Load the model
    cpu_model = CPUModel(model_file)
    time.sleep(2)  # Let memory usage stabilize

    log_and_print(f"RAM After Model Load: {get_ram_usage():.2f} MB")
    log_and_print(f"Unified Memory After Model Load: {get_unified_memory_usage():.2f} MB")

    # Select prompt
    input_text = loaded_prompts[0]
    log_and_print(f'Prompt:\n{input_text}')

    args = {
        'temperature': 0.5,
        'top_p': 0.3,
        'max_tokens': 500,
        'input_text': input_text
    }

    stop_event = threading.Event()
    start_time = time.time()

    # Start resource tracking in a separate thread
    monitor_thread = threading.Thread(target=track_resources, args=(
        stop_event, timestamps, cpu_usage_list, ram_usage_list, unified_mem_usage_list, start_time
    ))
    monitor_thread.start()

    log_and_print("\nStarting inference...\n")

    try:
        time.sleep(2)
        # Run model inference while resource tracking is active
        start_time = time.time()
        generated_text = cpu_model.get_text_response(args)
        time_taken = time.time() - start_time
        tokens = len(generated_text.split(' '))
        log_and_print(f'Time taken to run inference: {time_taken}s')
        log_and_print(f'Tokens generated: {tokens}')
        log_and_print(f'tokens/s: {tokens/time_taken}')
        time.sleep(2)
        del cpu_model
        gc.collect()
        time.sleep(2)

        stop_event.set()  # Stop tracking when inference is done
        monitor_thread.join()  # Wait for thread to finish

        log_and_print("\nGenerated Response:\n" + generated_text)

    except KeyboardInterrupt:
        stop_event.set()
        monitor_thread.join()
        log_and_print("\nKeyboard Interrupt detected. Saving graphs...")

    log_and_print("\nInference completed. Saving graphs...\n")

    # Plot RAM Usage
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, ram_usage_list, label="RAM Usage (MB)", color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Memory (MB)")
    plt.title(f"RAM Usage Over Time - {model_name}")
    plt.legend()
    plt.grid()
    ram_plot_filename = f"{model_name}/ram.png"
    plt.savefig(ram_plot_filename)
    log_and_print(f"Saved RAM usage graph as {ram_plot_filename}")

    # Plot Unified Memory Usage
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, unified_mem_usage_list, label="Unified Memory (MB)", color='green')
    plt.xlabel("Time (s)")
    plt.ylabel("Memory (MB)")
    plt.title(f"Unified Memory Usage Over Time - {model_name}")
    plt.legend()
    plt.grid()
    unified_mem_plot_filename = f"{model_name}/unified_mem.png"
    plt.savefig(unified_mem_plot_filename)
    log_and_print(f"Saved Unified Memory usage graph as {unified_mem_plot_filename}")

    # Plot CPU Usage
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, cpu_usage_list, label="CPU Usage (%)", color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title(f"CPU Usage Over Time - {model_name}")
    plt.legend()
    plt.grid()
    cpu_plot_filename = f"{model_name}/cpu.png"
    plt.savefig(cpu_plot_filename)
    log_and_print(f"Saved CPU usage graph as {cpu_plot_filename}")

    log_and_print("\nGraphs saved. Inference completed.")


for model_file in MODEL_FILES:
    monitor_inference(model_file)