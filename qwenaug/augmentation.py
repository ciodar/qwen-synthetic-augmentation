from PIL import Image
import torch
import pathlib as pl
import time

def insert_text(pipeline, input_path, text, prompt, output_dir):
    
    input_path = pl.Path(input_path)
    filename = input_path.name
    output_path = pl.Path(output_dir).joinpath(filename)
    
    image = Image.open(input_path).convert("RGB")
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

    output_image = output.images[0]
    output_image.save(output_path)
    print(f"Processed {filename} in {elapsed:.2f}s. Peak VRAM: {peak_vram:.2f} MiB. image saved in {output_path}")


def insert_image(pipeline, input_path, overlay_path, prompt, output_dir):

    input_path = pl.Path(input_path)
    filename = input_path.name
    output_path = pl.Path(output_dir).joinpath(filename)
    
    base_image = Image.open(input_path).convert("RGB")
    overlay_image = Image.open(overlay_path).convert("RGB")
    # mask_image=Image.open(pl.Path(input_path).parent.joinpath(input_path.stem + "_mask" + input_path.suffix))

    # Combine base_image and overlay_image
    width = base_image.width + overlay_image.width
    height = max(base_image.height, overlay_image.height)
    print(base_image.width, base_image.height, overlay_image.width, overlay_image.height, width, height)
    image = Image.new("RGB", (width, height))
    image.paste(base_image, (0, 0))
    # TODO: Add padding, center overlay
    image.paste(overlay_image, (base_image.width, 0))
    image.save(pl.Path(output_dir).joinpath(input_path.stem + "_overlay" + ".jpg"))

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
        "width": base_image.width,
        "height": base_image.height,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save(output_path)
        print("image saved at", output_path)