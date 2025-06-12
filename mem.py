import sys
import torch
import gc
import time
import pynvml

def format_bytes(bytes_value):
    """Formats bytes into a human-readable string (KB, MB, GB)."""
    if bytes_value is None:
        return "N/A"
    power = 1024
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while bytes_value >= power and n < len(power_labels) - 1:
        bytes_value /= power
        n += 1
    return f"{bytes_value:.2f} {power_labels[n]}B"

def get_size_mb(obj):
    """
    check object sizes (in mb) :d
    """
    bytes_size = sys.getsizeof(obj)
    mb_size = bytes_size / (1024 * 1024)  # Convert bytes to MB: bytes -> KB -> MB
    return f"{mb_size:.2f} MB"

def check_memory():
    """
    Check memory of all CUDA devices
    """
    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("No CUDA devices found.")
        return

    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024/1024/1024:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(i)/1024/1024/1024:.2f} GB")
        print(f"  Total: {torch.cuda.get_device_properties(i).total_memory/1024/1024/1024:.2f} GB")
        print()

def clear_all_cuda_memory(verbose=False):
    """
    Clear all CUDA memory
    """
    # Ensure all CUDA operations are complete
    torch.cuda.synchronize()
    
    # Empty the cache on all devices
    for device_id in range(torch.cuda.device_count()):
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    
    # Clear references to any tensors and force garbage collection
    gc.collect()
    
    # Optionally, reset the CUDA context (commented out as it's more drastic and may not always be necessary)
    # for device_id in range(torch.cuda.device_count()):
    #     torch.cuda.reset()
    if verbose:
        print("All CUDA memory cleared on all devices.")


def profile_memory(func, warmup = 3, runs = 10, *args, **kwargs):
    """
    Profile peak CUDA memory usage of a torch function. Uses warmup/multiple passes for more accuracy.

    Params
        @func: The function to test.
        @warmup: Number of warmup runs before timing.
        @runs: Number of timed runs.
        @args, kwarsg: Arguments to pass to the function
    
    Examples:
        profile_memory(np.diff, a = [0, 2, 5, 10, 12], n = 1)
    """
    for _ in range(warmup):
        func(*args, **kwargs)
        torch.cuda.synchronize()  # Make sure each warmup run finishes

    times = []
    peak_mems = []
    incd_mens = []

    for _ in range(runs):
        # Clear caches & reset memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure allocated memory before
        start_mem_bytes = torch.cuda.memory_allocated()
        
        # Start timing
        start_time = time.time()
        
        # Run the function (forward + backward)
        result = func(*args, **kwargs)
        
        # Synchronize to ensure all GPU work completes
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Measure memory usage after
        end_mem_bytes = torch.cuda.memory_allocated()
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        
        times.append(end_time - start_time)
        
        peak_mems.append(peak_mem_bytes)
        incd_mens.append(end_mem_bytes - start_mem_bytes)
        
    avg_time = sum(times)/len(times)
    avg_peak_mem = sum(peak_mems)/len(peak_mems)
    avg_incd_mem = sum(incd_mens)/len(incd_mens)

    return {
        "runs": runs,
        "average_time":  f"{avg_time:.8f}s",
        "average_peak_mem": f"{(avg_peak_mem/1e6):.4f}MB",
        "average_increase_mem_MB": f"{(avg_incd_mem/1e6):.4f}MB",
    }

def list_gpu_memory_usage(include_processes=False):
    """
    Lists memory usage for each detected NVIDIA GPU.

    Args:
        include_processes (bool): If True, also lists processes using GPU memory.
    """
    try:
        pynvml.nvmlInit()
        print("--- GPU Memory Usage ---")

        device_count = pynvml.nvmlDeviceGetCount()
        print(f"Found {device_count} GPU(s)")

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            total_memory = format_bytes(memory_info.total)
            used_memory = format_bytes(memory_info.used)
            free_memory = format_bytes(memory_info.free)

            print(f"\nGPU {i}: {gpu_name}")
            print(f"  Total Memory: {total_memory}")
            print(f"  Used Memory:  {used_memory}")
            print(f"  Free Memory:  {free_memory}")

            if include_processes:
                try:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    graphics_processes = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                    all_processes = processes + graphics_processes # Combine process lists

                    if all_processes:
                        print("  GPU Processes:")
                        # Sort processes by memory usage for better readability
                        all_processes.sort(key=lambda x: x.usedGpuMemory, reverse=True)
                        for proc in all_processes:
                            try:
                                process_name = pynvml.nvmlSystemGetProcessName(proc.pid)
                                print(f"    PID: {proc.pid:<6} | Memory: {format_bytes(proc.usedGpuMemory):<10} | Name: {process_name}")
                            except pynvml.NVMLError:
                                print(f"    PID: {proc.pid:<6} | Memory: {format_bytes(proc.usedGpuMemory):<10} | Name: N/A (Could not retrieve name)")
                    else:
                        print("  No processes using GPU memory.")
                except pynvml.NVMLError as err:
                    print(f"  Could not retrieve process information for GPU {i}: {err}")


    except pynvml.NVMLError as err:
        print(f"NVML Error: {err}")
        print("Please ensure that the NVIDIA drivers are installed and the NVML library is accessible.")
        if err.value == pynvml.NVML_ERROR_NOT_FOUND:
             print("NVML shared library not found. It might not be installed or not in the system's library path.")
        elif err.value == pynvml.NVML_ERROR_NO_PERMISSION:
            print("Insufficient permissions to access NVML. Try running with administrator/root privileges if necessary (use caution).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass # NVML might not have been initialized

def print_gpu_tensor_report():
    """
    Iterates through all Python objects, finds PyTorch CUDA tensors,
    and prints information about them. Helps identify unexpected lingering tensors.
    """
    print("\n--- GPU Tensor Report ---")
    total_mem_bytes = 0
    tensor_count = 0
    found_tensors = [] # Store basic info to potentially analyze duplicates/patterns

    for obj in gc.get_objects():
        try:
            # Check if it's a PyTorch tensor and resides on CUDA
            if torch.is_tensor(obj) and obj.is_cuda:
                tensor_count += 1
                # Calculate memory (storage size * element size)
                # storage().size() gives the total number of elements in the underlying storage
                # element_size() gives size of one element in bytes
                mem_bytes = obj.storage().size() * obj.element_size()
                total_mem_bytes += mem_bytes
                # Store/print info
                info = f"  Type: {type(obj).__name__}, Size: {obj.size()}, Dtype: {obj.dtype}, Mem: {mem_bytes / 1024**2:.2f} MB, Device: {obj.device}"
                print(info)
                found_tensors.append((obj.size(), obj.dtype, mem_bytes))

                # Optionally check for gradients if doing backprop-based saliency
                if hasattr(obj, 'grad') and obj.grad is not None and obj.grad.is_cuda:
                     grad_mem_bytes = obj.grad.storage().size() * obj.grad.element_size()
                     print(f"    Gradient - Size: {obj.grad.size()}, Dtype: {obj.grad.dtype}, Mem: {grad_mem_bytes / 1024**2:.2f} MB")

        except Exception as e:
            # Some objects might raise exceptions on access attempts
            # print(f"Error inspecting object: {e}")
            pass

    print(f"\nTotal {tensor_count} CUDA Tensors found.")
    print(f"Total Memory Held by Found Tensors: {total_mem_bytes / 1024**2:.2f} MB")
    # Compare with PyTorch's internal tracking
    print(f"torch.cuda.memory_allocated():     {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"torch.cuda.memory_reserved():      {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print("--- End Report ---\n")

