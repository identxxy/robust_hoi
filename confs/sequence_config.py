import os
import glob
import shutil

RUN_ON_SERVER = os.getenv("RUN_ON_SERVER", "").lower() == "true"
dataset = os.getenv("DATASET", "").lower()


def _detect_cuda_dir():
    """Auto-detect CUDA installation directory."""
    env = os.getenv("CUDA_HOME") or os.getenv("CUDA_DIR")
    if env and os.path.isdir(env):
        return env
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    for p in sorted(glob.glob("/usr/local/cuda*"), reverse=True):
        if os.path.isfile(os.path.join(p, "bin", "nvcc")):
            return p
    return ""


cuda_dir = _detect_cuda_dir()

if RUN_ON_SERVER:
    home_dir = "/data1/shibo/"
    conda_dir = "/home/shibo/.conda/"
else:
    home_dir = os.path.expanduser("~")
    conda_dir = os.getenv("CONDA_DIR", f"{home_dir}/.conda")

vggt_code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

body_models_dir = os.getenv("BODY_MODELS_DIR", f"{vggt_code_dir}/body_models")
gt_processed_dir = os.getenv("GT_PROCESSED_DIR", f"{vggt_code_dir}/HOD3D_v3/processed")
output_baseline_dir = os.getenv("OUTPUT_BASELINE_DIR", f"{vggt_code_dir}/output_baseline")

if dataset == "zed":
    dataset_dir = f"{home_dir}/Documents/dataset/ZED_wenxuan/"
    dataset_type = "zed"
    from confs.sequence_config_zed import sequences, sequence_name_list

elif dataset == "ho3d":
    dataset_dir = os.getenv("DATASET_DIR", "/data1/shibo/Documents/dataset/BundleSDF/HO3D_v3/train/")
    dataset_type = "ho3d"
    from confs.sequence_config_ho3d import sequences, sequence_name_list
else:
    raise ValueError(f"Please 'export DATASET=zed or ho3d' at first")           

