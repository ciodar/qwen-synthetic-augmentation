import csv, os
import argparse

from qwenaug.model import download_kontext_weights, load_flux_kontext, download_qwen_weights,  load_qwen_image_edit
from qwenaug.augmentation import insert_text, insert_image

from tqdm import tqdm
from pycocotools.coco import COCO
from diffusers.utils import load_image



def process_csv(pipeline, csv_path, ann_path="annotations/instances_train2017.json", output_dir="output"):
    coco = COCO(ann_path)
    images_dir = os.path.join(os.path.dirname(csv_path), "images")
    with open(csv_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in tqdm(reader):
            # Get fields from csv
            img_id=int(row.get("img_id"))
            overlay_text = row.get("overlay_text", "")
            prompt = row.get("prompt")
            
            # Get image from coco
            img = coco.imgs.get(img_id)

            # To load the mask, we should get the image category
            # cat_ids = your_image_category
            # anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            # anns = coco.loadAnns(anns_ids)
            # get one mask or concatenate all the masks for the category
            # mask = coco.annToMask(anns[0])

            try:
                image = load_image(img["flickr_url"])
            except:
                print(f"Failed to load image for img_id: {img_id} from {img["flickr_url"]}")
                continue
            output_path = os.path.join(output_dir, f'{img_id}.jpg')
            if not os.path.exists(output_path):
                if row.get("overlay_image", "") != "":
                    overlay_image = os.path.join(images_dir, row["overlay_image"])
                if overlay_text:
                    insert_text(pipeline, image, overlay_text, prompt, output_path)
                elif overlay_image:
                    insert_image(pipeline, image, overlay_image, prompt, output_path)


def main(args):
    print("Downloading model weights ...")
    download_qwen_weights()

    print("Loading model...")
    pipeline = load_qwen_image_edit()

    print(f"Starting augmentation process...")
    process_csv(pipeline, csv_path=args.csv_path,ann_path=args.ann_path,output_dir=args.output_dir)
    print(f"Augmented images saved in {args.output_dir}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV file for image augmentation.')
    parser.add_argument('--csv_path', type=str, required=False, default='examples/input.csv', help='Path to the input CSV file')
    parser.add_argument('--ann_path', type=str, required=False, default='examples/annotations/instances_train2017.json', help='Path to the input CSV file')
    parser.add_argument('--output_dir', type=str, required=False, default='output', help='Directory to save the output images')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.csv_path):
        print(f"Cannot find input file {args.csv_path}!")
    else:
        main(args)
