import os
import shutil
import pandas as pd
import random
import boto3
import zipfile
from io import BytesIO

# Paths and settings
bucket_name = 'deeplearning-mlops-demo'
zip_file_key = 'isic-2024-challenge.zip'
folder_to_extract = "train-image/image"
csv_filename = 'train-metadata.csv'
extract_to_folder = './extracted_files/'
output_folder = os.path.join(os.getcwd(), "classes")
num_images_to_save = 7

# Function to extract data from S3 and read CSV
def extract_data_and_read_csv(bucket, key, folder, csv_file, extract_to):
    s3 = boto3.client('s3')
    
    try:
        with BytesIO() as zip_buffer:
            s3.download_fileobj(bucket, key, zip_buffer)
            zip_buffer.seek(0)
            
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                all_files = zip_ref.namelist()
                files_in_folder = [f for f in all_files if f.startswith(folder)]
              
                for file in files_in_folder:
                    zip_ref.extract(file, path=extract_to)
                    
                with zip_ref.open(csv_file) as csv_fp:
                    # Set low_memory=False to avoid DtypeWarning
                    csv_data = pd.read_csv(csv_fp, low_memory=False)
                    print("CSV file read successfully.")
                    print(csv_data.head())
                
        if csv_data is None:
            print(f"CSV file '{csv_file}' not found in folder '{folder}'.")
        else:
            print("Data access complete.")
            return extract_to, csv_data
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return None, None

# Extract data and read CSV
extracted_path, df = extract_data_and_read_csv(bucket_name, zip_file_key, folder_to_extract, csv_filename, extract_to_folder)
if df is None:
    raise Exception("Failed to read CSV file.")

# Directories for the target classes
class_0_folder = os.path.join(output_folder, 'class_0')
class_1_folder = os.path.join(output_folder, 'class_1')

os.makedirs(class_0_folder, exist_ok=True)
os.makedirs(class_1_folder, exist_ok=True)

# Path to the extracted images
train_image_dir = os.path.join(extracted_path, folder_to_extract)

# Get a list of image files in the directory
image_files = set(os.listdir(train_image_dir))

# Move images to the respective directories based on the target values
for index, row in df.iterrows():
    image_name = row['isic_id'].strip() + '.jpg'
    target = row['target']
    
    src_path = os.path.join(train_image_dir, image_name)
    
    if target == 0:
        dst_path = os.path.join(class_0_folder, image_name)
    else:
        dst_path = os.path.join(class_1_folder, image_name)
    
    if image_name in image_files:
        shutil.move(src_path, dst_path)
    else:
        print(f"Image {image_name} not found in {train_image_dir}")

print("Images classified successfully.")

# Function to save random images
def save_random_images(class_folder, num_images, save_folder, class_label):
    images = [os.path.join(class_folder, img) for img in os.listdir(class_folder) if img.endswith('.jpg')]
    random_images = random.sample(images, min(num_images, len(images)))
    
    for jpg_file in random_images:
        jpg_path = os.path.join(save_folder, jpg_file)
        new_jpg_path = os.path.join(os.getcwd(), f"class_{class_label}")
    
    for img_path in random_images:
        shutil.copy(img_path, save_folder)

    print(f'Saved {len(random_images)} random images from class {class_label} to {save_folder}')

save_folder = os.getcwd()
# Save random images from class 0
save_random_images(class_0_folder, num_images_to_save, save_folder, 0)

# Save random images from class 1
save_random_images(class_1_folder, num_images_to_save, save_folder, 1)
 