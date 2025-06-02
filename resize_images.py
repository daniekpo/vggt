import os
from PIL import Image

def resize_and_save_images_from_paths(image_path_list, target_size=518, output_dir="data/images/resized/"):
    os.makedirs(output_dir, exist_ok=True)

    processed_files_count = 0
    failed_files_info = []

    print(f"Starting image processing. Output will be saved to: {os.path.abspath(output_dir)}")

    for image_path in image_path_list:
        img = Image.open(image_path)

        width, height = img.size

        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

        resized_img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

        base_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, base_filename)

        resized_img.save(output_path)
        processed_files_count += 1


    print("--- Processing Summary ---")
    print(f"Successfully processed and saved: {processed_files_count} image(s).")
    if failed_files_info:
        print(f"Failed to process: {len(failed_files_info)} image(s):")
        for item in failed_files_info:
            print(f"  - Path: {item['path']}, Reason: {item['reason']}")
    print(f"Resized images are in: {os.path.abspath(output_dir)}")