# utils/model_loader.py

import os


def load_model_list(models_dir="models"):
    """
    models 폴더에서 .pt 파일을 찾아 정렬된 리스트로 반환.
    Predict 페이지에서 사용.
    """
    if not os.path.exists(models_dir):
        return []

    files = os.listdir(models_dir)
    pt_files = [f for f in files if f.lower().endswith(".pt")]
    pt_files.sort()
    return pt_files
