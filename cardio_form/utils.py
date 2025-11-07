import os 

def get_unique_parts(file_paths):
    # Strip directory paths and file extensions
    base_names = [os.path.splitext(os.path.dirname(path))[0] for path in file_paths]
    
    # Find the common prefix
    common_prefix = os.path.commonprefix(base_names)
    return common_prefix

def check_file_exists(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
