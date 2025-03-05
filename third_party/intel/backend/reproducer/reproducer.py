import ctypes
import numbers
import os
import torch
import shutil

from triton.runtime.cache import get_cache_manager


def save_raw_tensor(tensor, file_path):
    tensor = tensor.cpu().contiguous()
    data_type = ctypes.c_char * tensor.numel() * tensor.element_size()
    data = data_type.from_address(tensor.data_ptr())
    with open(file_path, 'wb') as f:
        f.write(data)


def make_tensor_loop(tensor, name, shape, idx, lines, sfx):
    if idx == len(shape):
        lines.append(
            f"{sfx}size_t idx = i0 * {name}Strides[0]{''.join(f' + i{i} * {name}Strides[{i}]' for i in range(1, idx))};")
        if (hasattr(tensor, "compare_with_rtol")
                or hasattr(tensor, "compare_with_atol")
                or hasattr(tensor, "compare_with_equal_nan")):
            rtol = getattr(tensor, "compare_with_rtol", 1e-05)
            atol = getattr(tensor, "compare_with_atol", 1e-08)
            equal_nan = str(getattr(tensor, "compare_with_equal_nan", False)).lower()
            lines.append(f"{sfx}if (!isClose({name}[idx], {name}Ref[idx], {rtol}, {atol}, {equal_nan})) {{")
        else:
            lines.append(f"{sfx}if ({name}[idx] != {name}Ref[idx]) {{")
        lines.append(f"{sfx}  std::string err = \"Mismatch at index \";")
        lines.append(f"{sfx}  err.append(std::to_string(idx));")
        lines.append(f"{sfx}  err.append(\": \");")
        lines.append(f"{sfx}  err.append(std::to_string({name}[idx]));")
        lines.append(f"{sfx}  err.append(\" != \");")
        lines.append(f"{sfx}  err.append(std::to_string({name}Ref[idx]));")
        lines.append(f"{sfx}  throw std::runtime_error(err.c_str());")
        lines.append(f"{sfx}}}")
    else:
        lines.append(f"{sfx}for (size_t i{idx} = 0; i{idx} < {shape[idx]}ULL; i{idx}++) {{")
        make_tensor_loop(tensor, name, shape, idx + 1, lines, sfx + "  ")
        lines.append(f"{sfx}}}")


def create_reproducer(dir_path, args, constants, signature):
    data_dir_path = os.path.join(dir_path, 'data')
    os.makedirs(data_dir_path, exist_ok=True)
    is_native = os.getenv("TRITON_XPU_GEN_NATIVE_CODE", False)
    # 3: stream, 4: function, 5: packed kernel metadata, 6: launch_metadata, 7: launch_enter_hook, 8: launch_exit_hook
    assert type(args[5]).__name__ == "KernelMetadata"
    meta = args[5]
    args_dict = {"gridX": args[0], "gridY": args[1], "gridZ": args[2], "num_warps": meta.num_warps,
                 "threads_per_warp": meta.threads_per_warp, "shared_memory": meta.shared, "kernel_name": meta.name,
                 "build_flags": meta.build_flags, "arguments": []}
    scalars_cnt = 0
    tensors = {}
    tensors_ref = []
    tensors_ref_cmp = []
    set_args = []
    for karg_cnt, ((sig_name, sig_type), arg) in enumerate(zip(signature.items(), args[9:])):
        if isinstance(arg, torch.Tensor):
            name = f"tensor{len(tensors)}"
            ctype = f"tt_{sig_type[1:]}"
            is_float = "f" in sig_type
            tensors[name] = ctype
            set_args.append(f"{karg_cnt}, static_cast<void *>({name}Dev.ptr)")
            save_raw_tensor(arg, os.path.join(data_dir_path, f"{name}.bin"))
            if hasattr(arg, "compare_with"):
                tensors_ref.append(name)
                tensors_ref_cmp.append(f"  {{ // Compare {name} with the reference")
                tensors_ref_cmp.append(f"    auto {name}Ref = readFile<{ctype}>(\"{name}_ref.bin\");")
                tensors_ref_cmp.append(f"    assert({name}.size() == {name}Ref.size());")
                tensors_ref_cmp.append(
                    f"    size_t {name}Strides[{len(arg.shape)}] = {{{', '.join([f'{a}ULL' for a in arg.stride()])}}};")
                make_tensor_loop(arg if is_float else None, name, arg.shape, 0, tensors_ref_cmp, "    ")
                tensors_ref_cmp.append("  }")
                save_raw_tensor(arg.compare_with, os.path.join(data_dir_path, f"{name}_ref.bin"))
            new_arg = {
                "name": name, "type": "tensor", "dtype": str(arg.dtype), "ctype":
                    sig_type, "shape": list(arg.shape), "strides": list(arg.stride())
            }
            args_dict["arguments"].append(new_arg)
        if isinstance(arg, numbers.Number) and not (karg_cnt,) in constants.keys():
            set_args.append(f"{karg_cnt}, static_cast<tt_{sig_type}>({arg})")
            new_arg = {
                "name": f"scalar{scalars_cnt}", "type": "scalar", "value": arg, "ctype": sig_type
            }
            args_dict["arguments"].append(new_arg)
            scalars_cnt += 1

    if meta.shared:
        set_args.append(f"{len(set_args)}, sycl::local_accessor<int8_t>({meta.shared}, h)")

    # Dump argument info as a JSON file
    with open(os.path.join(data_dir_path, "args_info.json"), "w") as f:
        import json
        json.dump(args_dict, f, indent=4)

    # Copy the kernel and IRs from the cache
    cache = get_cache_manager(meta.hash)
    for name, path in cache.get_group(f"{meta.name}.json").items():
        shutil.copy(path, os.path.join(data_dir_path, name))

    # Create the native launcher code
    reproducer_dir = os.path.dirname(__file__)
    shutil.copy(os.path.join(reproducer_dir, "CMakeLists.txt"), dir_path)
    shutil.copy(os.path.join(reproducer_dir, "KernelLauncher.cpp"), dir_path)
    shutil.copy(os.path.join(reproducer_dir, "launch.sh"), dir_path)
    join_lines = "\n".join
    with open(os.path.join(dir_path, "KernelLauncher.cpp"), "a") as f:
        f.write(f"""
int main(const int argc, const char **argv) {{
  std::filesystem::current_path("data");
{join_lines([f'  auto {n} = readFile<{t}>("{n}.bin");' for n, t in tensors.items()])}

  {{ // Launch the kernel
    const sycl::device device = findDevice(argc == 1 ? "" : argv[1]);
    sycl::queue queue(device);
    Launcher launcher(queue, "{meta.name}.{'xebin' if is_native else 'spv'}", "{meta.name}", {'true' if is_native else 'false'});
{join_lines([f'    auto {n}Dev = launcher.copyFrom({n});' for n in tensors.keys()])}
    queue.wait_and_throw();
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    auto event = launcher.submit({args[0]}, {args[1]}, {args[2]}, {meta.num_warps}, {meta.threads_per_warp}, [&](sycl::handler &h) {{
{join_lines([f'      h.set_arg({a});' for a in set_args])}
    }});
    event.wait_and_throw();
    auto end = high_resolution_clock::now();
    std::cout << "Kernel execution time: "
              << duration_cast<milliseconds>(end - start).count() << " ms"
              << std::endl;
{join_lines([f'    launcher.copyTo({n}Dev, {n});' for n in tensors_ref])}
    queue.wait_and_throw();
  }}
{join_lines(tensors_ref_cmp)}
  return 0;
}}
""");
