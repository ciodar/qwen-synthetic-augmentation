import csv, os
import argparse

from qwenaug.model import download_kontext_weights, load_flux_kontext
from qwenaug.augmentation import insert_text, insert_image

from tqdm import tqdm


def process_csv(pipeline, csv_path, output_dir):
    images_dir = os.path.join(os.path.dirname(csv_path), "images")
    with open(csv_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in tqdm(reader):
            input_image = os.path.join(images_dir, row["input_image"])
            overlay_text = row.get("overlay_text", "")
            prompt = row.get("prompt")
            if row.get("overlay_image", "") != "":
                overlay_image = os.path.join(images_dir, row["overlay_image"])
            if overlay_text:
                insert_text(pipeline, input_image, overlay_text, prompt, output_dir)
            elif overlay_image:
                insert_image(pipeline, input_image, overlay_image, prompt, output_dir)


def main(args):
    print("Downloading model weights ...")
    download_kontext_weights()

    print("Loading model...")
    pipeline = load_flux_kontext()

    print(f"Starting augmentation process...")
    process_csv(pipeline, args.csv_path, args.output_dir)
    print(f"Augmented images saved in {args.output_dir}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV file for image augmentation.')
    parser.add_argument('--csv_path', type=str, required=False, default='input.csv', help='Path to the input CSV file')
    parser.add_argument('--output_dir', type=str, required=False, default='output', help='Directory to save the output images')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.csv_path):
        print(f"Cannot find input file {args.csv_path}!")
    else:
        main(args)
