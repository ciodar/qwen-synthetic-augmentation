from PIL import Image
import torch
import os 
import pathlib as pl
import time

def insert_text(pipeline, image, text, prompt, output_path):
    
    inputs = {
        "image": image,
        "prompt": prompt.format(text),
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
    }

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        output = pipeline(**inputs)

    elapsed = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    print(f"Processed {os.path.basename(output_path)} in {elapsed:.2f}s. Peak VRAM: {peak_vram:.2f} MiB. image saved in {output_path}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    output_image = output.images[0]
    output_image.save(output_path)
    
    


def insert_image(pipeline, image, overlay_path, prompt, output_path):
    base_image = image
    overlay_image = Image.open(overlay_path).convert("RGB")
    # mask_image=Image.open(pl.Path(input_path).parent.joinpath(input_path.stem + "_mask" + input_path.suffix))

    # Combine base_image and overlay_image
    width = base_image.width + overlay_image.width
    height = max(base_image.height, overlay_image.height)
    image = Image.new("RGB", (width, height))
    image.paste(base_image, (0, 0))
    # TODO: Add padding, center overlay
    image.paste(overlay_image, (base_image.width, 0))
    # image.save(pl.Path(output_dir).joinpath(input_path.stem + "_overlay" + ".jpg"))

    # mask_image = Image.open(overlay_path).convert("RGB")
    print(prompt)
    # Compose base image and overlay

    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
        "width": image.width,
        "height": image.height,
    }

    start_time = time.time()

    with torch.inference_mode():
        output = pipeline(**inputs)
    
    elapsed = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    print(f"Processed {os.path.basename(output_path)} in {elapsed:.2f}s. Peak VRAM: {peak_vram:.2f} MiB. image saved in {output_path}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    output_image = output.images[0]
    output_image = output_image.crop(box=(0,0,base_image.width, base_image.height))
    output_image.save(output_path)