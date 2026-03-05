import os
RUN_ON_SERVER = os.getenv("RUN_ON_SERVER", "").lower() == "true"
dataset = os.getenv("DATASET", "").lower()

if RUN_ON_SERVER:
    home_dir = "/data1/shibo/"
    conda_dir = "/home/shibo/.conda/"
else:
    home_dir = os.path.expanduser("~")
    conda_dir = f"{home_dir}/miniconda3"

if dataset == "zed":
    dataset_dir = f"{home_dir}/Documents/dataset/ZED_wenxuan/"
    dataset_type = "zed"

    sequence_name_list = [
        ###### in-the-wild captured by ZED ##########
        "CUB1",
        "CUB2",
        "DUC1",
        "DUC2",
        "TC3",
        "TC4",
        "WC3",
        "WC4", 
    ]
    from confs.sequence_config_zed import sequences
elif dataset == "ho3d":
    dataset_dir = f"{home_dir}/Documents/dataset/BundleSDF/HO3D_v3/train/"
    dataset_type = "ho3d"
    sequence_name_list = [
        "ABF12",
        "ABF14",
        "GPMF12",
        "GPMF14",    
        "MC1",
        "MC4",
        "MDF12",
        "MDF14",
        "ShSu10",
        "ShSu14",
        "SM2",
        "SM4",
        "SMu1",
        "SMu40",
        "BB12",
        "BB13", 
        "GSF12",
        "GSF13",       
    ]
    from confs.sequence_config_ho3d import sequences
else:
    raise ValueError(f"Please 'export DATASET=zed or ho3d' at first")           

vggt_code_dir = f"{home_dir.rstrip('/')}/Documents/project/vggt"
