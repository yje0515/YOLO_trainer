import os

def load_model_list(models_dir="models"):
    """models 폴더에서 .pt 파일 자동 탐색하여 리스트 반환"""
    if not os.path.exists(models_dir):
        return []

    files = os.listdir(models_dir)
    pt_files = [f for f in files if f.lower().endswith(".pt")]
    pt_files.sort()  # 정렬
    return pt_files
