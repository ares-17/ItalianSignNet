import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import random
import logging
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime

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

def load_config():
    load_dotenv()
    base_dir = os.getenv("BASE_DIR_DATASET")
    if not base_dir:
        raise ValueError("BASE_DIR_DATASET non definita nel file .env")
    return Path(base_dir)

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
            logger.warning(f"Immagine non trovata o illeggibile: {img_path}")
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
                new_rows.append(new_row)
        else:
            logger.error(f"Errore nel salvataggio di: {new_path}")

    return new_rows

def main():
    logger = define_logger()
    base_dir = load_config()
    train_dir = base_dir / "train"
    metadata_path = base_dir / "metadata.parquet"

    random.seed(SEED)
    np.random.seed(SEED)

    logger.info(f"Caricamento metadata da {metadata_path}")
    metadata = pd.read_parquet(metadata_path)

    all_images = get_image_list(train_dir)
    logger.info(f"Trovate {len(all_images)} immagini totali")

    photo_transform, atmospheric_transform, occlusion_transform = get_transforms()
    photo_imgs, atm_imgs, occlusion_imgs = sample_images(all_images)

    new_metadata_rows = []
    new_metadata_rows += apply_and_save_transform(photo_imgs, photo_transform, "Photometric", logger, metadata)
    new_metadata_rows += apply_and_save_transform(atm_imgs, atmospheric_transform, "Atmospheric", logger, metadata)
    new_metadata_rows += apply_and_save_transform(occlusion_imgs, occlusion_transform, "Occlusion", logger, metadata)

    if new_metadata_rows:
        logger.info(f"Aggiunte {len(new_metadata_rows)} nuove righe al metadata")
        augmented_df = pd.DataFrame(new_metadata_rows)
        updated_metadata = pd.concat([metadata, augmented_df], ignore_index=True)
        updated_metadata.to_parquet(metadata_path, index=False)
        logger.info(f"Metadata aggiornato salvato in {metadata_path}")
    else:
        logger.info("Nessuna nuova immagine generata.")

if __name__ == "__main__":
    main()
