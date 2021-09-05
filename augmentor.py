import os
import glob
import argparse
import random
from pathlib import Path
from pprint import pprint

import imagehash
from PIL import Image, ImageOps
import progressbar
from multiprocessing.pool import ThreadPool
from termcolor import colored


stats = {
    'images_collected': 0,
    'images_duplicated': 0,
}
hashes = []  # Image hashes
hash_size = 15


def get_max_window_size(image: Image, ratio: float):
    """ Returns maximum size (width, height) of inscribed rectangle with specific aspect ratio """
    width, height = image.size
    current_ratio = width / height
    if ratio == current_ratio:
        return width, height
    elif ratio > current_ratio:
        return width, round(width / ratio)
    elif ratio < current_ratio:
        return round(height * ratio), height


def print_config():
    """ Print setup """
    for arg in vars(config):
        print("{0:<16} {1:<16}".format(arg, str(getattr(config, arg))))
    print("---------------------------------------")
    # Check config
    assert(config.crops > 0)
    assert(0 <= config.scale_min <= config.scale_max <= 1)


def process_image(file: str):
    """ Augmentation process of single image """
    global stats

    # Load image
    try:
        image_original = Image.open(file).convert('RGB')
    except Exception as e:
        print(file, colored("ERROR", 'red'), colored("Loading image failed", 'yellow'), e)
        return

    # Skip images smaller then required size
    if image_original.width < config.width or image_original.height < config.height:
        print(file, colored("SKIPPED (too small)", 'red'))
        return

    # Skip duplicity image
    if config.dedupe_input:
        image_hash = imagehash.average_hash(image_original, hash_size=hash_size)
        if image_hash in hashes:
            print(file, colored("DUPLICATE ", 'red'))
            stats['images_duplicated'] += 1
            return
        hashes.append(image_hash)

    stem = Path(file).stem
    window_max = get_max_window_size(image_original, config.width / config.height)

    variants = {}  # {filename : PIL.Image}
    for n in range(1, config.crops + 1):
        image = image_original.copy()

        # Autocontrast with `config.autocontrast` probability
        if random.random() < config.autocontrast:
            cutoff = random.uniform(config.cutoff_min, config.cutoff_max)
            image = ImageOps.autocontrast(image, cutoff=cutoff)
        else:
            cutoff = False

        # Random window
        scale_min = config.width / window_max[0]
        scale = random.uniform(max(scale_min, config.scale_min), config.scale_max)  # Random scale
        window_width = round(scale * window_max[0])
        window_height = round(scale * window_max[1])
        left = random.randint(0, image.width - window_width)
        top = random.randint(0, image.height - window_height)
        crop_box = (left, top, left + window_width, top + window_height)

        # todo Zkusit jestli resize+box ned stejný výsledej
        image_cropped = image.crop(crop_box).resize((config.width, config.height), resample=3, box=None, reducing_gap=None)

        # centering = (random.uniform(0, 1), random.uniform(0, 1))
        # image_cropped = ImageOps.fit(image, size=(config.width, config.height), method=Image.ANTIALIAS, bleed=0, centering=centering)
        if config.randomize:
            outfile = f"{config.output_dir}/{n}-{stem}--ac{cutoff:.2f}.{config.format}"
        else:
            outfile = f"{config.output_dir}/{stem}-{n}--ac{cutoff:.2f}.{config.format}"
        variants[outfile] = image_cropped

    # Save varinats (skip duplicities)
    variants_hashes = []
    variants_saved = 0
    for file_name, image in variants.items():
        variant_hash = imagehash.average_hash(image, hash_size=hash_size)
        if variant_hash in variants_hashes:
            continue
        variants_hashes.append(variant_hash)
        image.save(file_name, quality=config.jpg_quality, subsampling=0)
        variants_saved += 1

    stats['images_collected'] += variants_saved
    print(file, colored("OK", 'green'), colored(str(variants_saved), 'cyan'))


def run():
    # Print setup
    print_config()

    # Get list of files
    # @todo Musi se prochazet i JPG jpeg JPEG png PNG apod.
    if config.recursive:
        files = glob.glob(config.input_dir + '/**/*.jpg', recursive=True)
    else:
        files = glob.glob(config.input_dir + '/*.jpg')

    images_found = len(files)
    if config.limit:
        try:
            files = random.sample(files, config.limit)
        except Exception:
            pass  # Images count is less then limit. Using all images found.

    print("Input images found:         ", images_found)
    print("Input images selected:      ", len(files))
    print("Expected output images (max):     ", len(files) * config.crops)

    # If dry run or no images found, exit.
    if config.dry or len(files) <= 0:
        exit()

    # Process images
    os.makedirs(config.output_dir, exist_ok=True)

    print(len(files))
    results = ThreadPool(config.threads).imap_unordered(process_image, files)
    for result in progressbar.progressbar(results, max_value=len(files), redirect_stdout=True):
        continue

    pprint(stats)
    print("Total collected images:", stats['images_collected'])


# Parse CLI arguments
parser = argparse.ArgumentParser()

# General options
parser.add_argument('-i', '--input-dir', type=str, default='./', help='Input directory with images (default: %(default)s)')
parser.add_argument('-o', '--output-dir', type=str, default='./augmentor-output/', help='Output directory. It will be created if it does not exist. (default: %(default)s)')
parser.add_argument("--recursive", help="Parse 'input_dir' recursively including all subfolders", action='store_true')
parser.add_argument("--limit", help="Process only 'limit' randomly selected samples from 'input-dir'", type=int)
parser.add_argument("--dry", help="Dry run. Only show config and calculate expected images count after augmentation.", action='store_true')
parser.add_argument("--threads", help="Threads count. (default: Number physical CPU cores)", type=int, default=6)
parser.add_argument("--dedupe-input", help="Deduplicate images on input", action='store_true')

# Output image format
parser.add_argument('--width', type=int, default=1024, help='Final output image width (default: %(default)s)')
parser.add_argument('--height', type=int, default=1024, help='Final output image height (default: %(default)s)')
parser.add_argument("--format", help="Ouput image format", type=str, default='jpg', choices=['jpg', 'png'])
parser.add_argument("--jpg-quality", help="JPEG quality (default: %(default)s)", type=int, default=100)
parser.add_argument("--randomize", help="Random image preffix to randomize order", action='store_true')

# @todo Implement --size
# @todo Implement --png-quality
# parser.add_argument('--size', type=str, default='1024x1024', help="Final output image size (width x height) e.g. '1280x768', '512x512' (default: %(default)s)")
# parser.add_argument("--png-quality", help="JPEG quality", type=int, default=100)

# Augmenation options
parser.add_argument('--crops', type=int, default=1, help='Maximum number of images to generate from a single image (default: %(default)s)')
parser.add_argument('--scale-min', type=float, default=0.8, help='Minimum zoom-out factor [0..1] (default: %(default)s)')
parser.add_argument('--scale-max', type=float, default=1.0, help='Maximum zoom-out factor [0..1] (default: %(default)s)')
parser.add_argument("--autocontrast", help="Autocontrast probability [0..1] (default: %(default)s)", default=0, type=float)
parser.add_argument("--cutoff-min", help="Minimum autocontrast cutoff. Minimum percent to cut off from the histogram [0..100] (default: %(default)s)", default=0, type=float)
parser.add_argument("--cutoff-max", help="Maximum autocontrast cutoff. Maximum percent to cut off from the histogram [0..100] (default: %(default)s)", default=1, type=float)


# parser.add_argument("--mirror", help="Mirror (horizontal flip) probability", type=int, default=0)
# parser.add_argument("--flip", help="Vertical flip probability", type=int, default=0)
# parser.add_argument("--rotate", help="Rotate 90 degrees probability", type=int, default=0)

# Run
config = parser.parse_args()
run()
