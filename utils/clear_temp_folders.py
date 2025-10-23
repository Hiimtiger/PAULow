import os
import shutil

def clear_temp_files():
    folders_to_clear = ['utils/temp_files']

    for folder in folders_to_clear:
        folder_path = os.path.join(os.getcwd(), folder)  # Ensure absolute path

        if os.path.exists(folder_path):
            for root, dirs, files in os.walk(folder_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif','.pth')): 
                        os.remove(file_path)

    print("[INFO] Clearing Images Complete.")
#clear_images_in_folder()