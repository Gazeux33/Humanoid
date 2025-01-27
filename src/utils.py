import os

def get_last_save_path(path:str)-> str:
    return os.path.join(path,max(os.listdir(path), key=lambda x: int(x.split('_')[1].split('.')[0])).replace('.zip', ''))