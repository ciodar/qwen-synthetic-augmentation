from PIL import Image
import torch
import os
import pathlib as pl

def insert_text(pipeline, input_path, text, output_dir):
    prompt = f'Add the text "{text}" to the image.'
    input_path = pl.Path(input_path)
    filename = input_path.name
    output_path = pl.Path.joinpath(output_dir, filename)
    
    image = Image.open(input_path).convert("RGB")
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save(output_path)
        print("image saved at", output_path)

def insert_image(pipeline, input_path, overlay_path, output_dir):
    prompt = f'Insert the object on the right in the image.'
    input_path = pl.Path(input_path)
    filename = input_path.name
    output_path = pl.Path.joinpath(output_dir, filename)
    
    base_image = Image.open(input_path).convert("RGB")
    overlay_image = Image.open(input_path).convert("RGB")

    # Combine base_image and overlay_image
    width = base_image.width + overlay_image.width
    height = max(base_image.height, overlay_image.height)
    image = Image.new("RGB", (width, height))
    image.paste(base_image, (0, 0))
    # TODO: Add padding, center overlay
    image.paste(overlay_image, (base_image.width, 0))

    # Compose base image and overlay
    image = None

    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save(output_path)
        print("image saved at", output_path)