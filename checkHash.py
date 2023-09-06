import os
import filecmp
import hashlib

def get_file_hash(file_path, block_size=65536):
    """Calculate SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        block = f.read(block_size)
        while block:
            hasher.update(block)
            block = f.read(block_size)
    return hasher.hexdigest()

def are_folders_equal(dir1, dir2):
    """Recursively compare two directories."""
    
    # 1. Check if both directories exist
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        return False
    
    # 2. Compare directory content (files and subdirectories)
    compared_dirs = filecmp.dircmp(dir1, dir2)
    if compared_dirs.left_only or compared_dirs.right_only:
        return False
    
    # 3. Check file contents
    for common_file in compared_dirs.common_files:
        file1 = os.path.join(dir1, common_file)
        file2 = os.path.join(dir2, common_file)
        if get_file_hash(file1) != get_file_hash(file2):
            return False
    
    # 4. Recursively check subdirectories
    for common_dir in compared_dirs.common_dirs:
        subdir1 = os.path.join(dir1, common_dir)
        subdir2 = os.path.join(dir2, common_dir)
        if not are_folders_equal(subdir1, subdir2):
            return False
    
    return True

if __name__ == "__main__":
    dir1 = "models"
    dir2 = "models2"

    if are_folders_equal(dir1, dir2):
        print("Both folders are the same.")
    else:
        print("The folders are different.")

