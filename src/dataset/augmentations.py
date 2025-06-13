import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import random
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import glob

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR") or ''
SEED = 42
PHOTO_PERCENT = 0.10
ATMOSPHERIC_PERCENT = 0.15
OCCLUSION_PERCENT = 0.10
EXTS = [".jpg", ".png", ".jpeg"]

def define_logger():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f"augmentations_{timestamp_str}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def get_transforms():
    photo_transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=[-0.5, 0.2],
                contrast_limit=[-0.2, 0.2],
                brightness_by_max=True,
                ensure_safe_range=False,
                p=1.0
            ),
            A.ChromaticAberration(
                primary_distortion_limit=[-0.5, 0.5],
                secondary_distortion_limit=[-0.3, 0.3],
                mode="random",
                interpolation=cv2.INTER_LINEAR,
                p=1.0
            )
        ])
    ], p=1)

    atmospheric_transform = A.OneOf([
        A.RandomRain(
            slant_range=[-15, 15],
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3,
            brightness_coefficient=0.7,
            rain_type="default",
            p=0.6
        ),
        A.RandomFog(
            fog_coef_lower=0.3,
            fog_coef_upper=0.8,
            alpha_coef=0.08,
            p=0.2
        ),
        A.Spatter(
            mean=[0.25, 0.25],
            std=[0.2, 0.2],
            gauss_sigma=[1, 2],
            cutout_threshold=[0.58, 0.68],
            intensity=[0.6, 0.8],
            mode="mud",
            p=0.2
        )
    ], p=1)

    occlusion_transform = A.GridDropout(
        holes_number_xy=[1, 3],
        ratio=0.25,
        random_offset=True,
        fill="random_uniform",
        p=1
    )

    return photo_transform, atmospheric_transform, occlusion_transform

def get_image_list(train_dir):
    return [p for p in train_dir.glob("*/*") if p.suffix.lower() in EXTS]

def sample_images(image_paths):
    total_images = len(image_paths)
    photo_imgs = random.sample(image_paths, int(PHOTO_PERCENT * total_images))
    remaining = list(set(image_paths) - set(photo_imgs))
    atm_imgs = random.sample(remaining, int(ATMOSPHERIC_PERCENT * total_images))
    remaining = list(set(remaining) - set(atm_imgs))
    occlusion_imgs = random.sample(remaining, int(OCCLUSION_PERCENT * total_images))
    return photo_imgs, atm_imgs, occlusion_imgs

def apply_and_save_transform(images, transform, label, logger, metadata_df):
    new_rows = []

    for img_path in tqdm(images, desc=f"Applying {label}"):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Image not found: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = transform(image=img)['image']
        aug_img = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        base_name = img_path.stem
        ext = img_path.suffix
        rand_suffix = str(np.random.randint(10000, 99999))
        new_filename = f"{base_name}_{rand_suffix}{ext}"

        new_path = img_path.parent / new_filename
        success = cv2.imwrite(str(new_path), aug_img)

        if success:
            original_row = metadata_df[metadata_df["filename"] == img_path.name]
            if not original_row.empty:
                new_row = original_row.iloc[0].copy()
                new_row["filename"] = new_filename
                new_row["generated"] = True
                new_rows.append(new_row)
        else:
            logger.error(f"Error during save: {new_path}")

    return new_rows

def save_new_metadata_rows(new_metadata_rows: list, metadata: pd.DataFrame, logger: logging.Logger, metadata_path: str):
    if new_metadata_rows:
        logger.info(f"Aggiunte {len(new_metadata_rows)} nuove righe al metadata")
        augmented_df = pd.DataFrame(new_metadata_rows)
        updated_metadata = pd.concat([metadata, augmented_df], ignore_index=True)
        updated_metadata.to_parquet(metadata_path, index=False)
        logger.info(f"Metadata aggiornato salvato in {metadata_path}")
    else:
        logger.info("Nessuna nuova immagine generata.")

def apply_trasform_and_save_metadata(all_images: list, logger: logging.Logger, metadata: pd.DataFrame, metadata_path: str):
    photo_transform, atmospheric_transform, occlusion_transform = get_transforms()
    photo_imgs, atm_imgs, occlusion_imgs = sample_images(all_images)

    new_metadata_rows = []
    new_metadata_rows += apply_and_save_transform(photo_imgs, photo_transform, "Photometric", logger, metadata)
    new_metadata_rows += apply_and_save_transform(atm_imgs, atmospheric_transform, "Atmospheric", logger, metadata)
    new_metadata_rows += apply_and_save_transform(occlusion_imgs, occlusion_transform, "Occlusion", logger, metadata)

    save_new_metadata_rows(new_metadata_rows, metadata, logger, metadata_path)

def get_input_dataset_or_latest(dataset_name=None) -> str:
    base_dir_dataset = Path(BASE_DIR, 'src', 'dataset', 'artifacts')

    if dataset_name:
        return os.path.join(base_dir_dataset, dataset_name)

    dataset_folders = glob.glob(os.path.join(base_dir_dataset, "dataset_*"))
    if not dataset_folders:
        raise FileNotFoundError("Nessun dataset trovato")

    dataset_folders.sort(reverse=True)
    return dataset_folders[0]

def main(dataset_name=None):
    logger = define_logger()

    dataset = get_input_dataset_or_latest(dataset_name)
    train_dir = Path(os.path.join(dataset, "train"))
    metadata_path = os.path.join(dataset, "metadata.parquet")

    random.seed(SEED)
    np.random.seed(SEED)

    logger.info(f"Loading metadata at {metadata_path}")
    metadata = pd.read_parquet(metadata_path)
    metadata["generated"] = False

    all_images = get_image_list(train_dir)
    logger.info(f"Found {len(all_images)} total images")

    apply_trasform_and_save_metadata(all_images, logger, metadata, metadata_path)

if __name__ == "__main__":
    main()
