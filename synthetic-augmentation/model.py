import os
from PIL import Image
import torch

def load_qwen_image_edit():
    from diffusers import QwenImageEditPipeline, AutoencoderKL, QwenImageTransformer2DModel, GGUFQuantizationConfig
    from diffusers.hooks import apply_group_offloading
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    # The original pipeline would be simply
    # pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")

    # Since I'm GPU-poor, I'll load a heavily quantized model from QuantStack/Qwen-Image-Edit-GGUF
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    # Load quantized Transformer
    transformer = QwenImageTransformer2DModel.from_single_file(
        pretrained_model_link_or_path_or_dict='weights/transformer/Qwen_Image_Edit-Q4_0.gguf',
        quantization_config=quantization_config,
        config="weights/transformer/config.json",
        dtype=torch.bfloat16
    )
    transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    print("Loaded transformer!")
    # Load Text Encoder
    # GGUF still not implemented see https://github.com/huggingface/transformers/issues/40049
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        #'weights/text_encoder', 
        #quantization_config=quantization_config,
        dtype=torch.bfloat16
    )
    # Load image encoder
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("loaded processor")
    # Load VAE
    # vae = AutoencoderKL.from_single_file('weights/vae/Qwen_Image-VAE.safetensors',
    #                                      config='weights/vae/config.json')

    pipeline = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        transformer=transformer,
        # vae=vae,
        text_encoder=text_encoder,
        processor=processor,
        dtype=torch.bfloat16
    )

    print("pipeline loaded")
    # Apply group offloading
    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")
    pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", stream="True")
    pipeline.vae.enable_group_offload(onload_device=onload_device, offload_type="leaf_level", stream="True")

    # Use the apply_group_offloading method for other model components
    apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2, stream="True")

    pipeline.to("cuda")
    # pipeline.set_progress_bar_config(disable=None)
    
   
    return pipeline

    

if __name__ == "__main__":
    # Example usage:
    pipeline = load_qwen_image_edit()
    image = Image.open("examples/input.jpg").convert("RGB")
    prompt = "Change the rabbit's color to purple, with a flash light background."
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 20,
    }

    if pipeline is not None:
        print("Qwen-Image Diffusers model loaded successfully!")
        # You can now use the 'image_pipeline' object to generate images.
        # For example:
        # image = image_pipeline("A photo of a cat wearing a hat")
        # image.save("cat_with_hat.png")
    else:
        print("Failed to load Qwen-Image Diffusers model.")

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save("output_image_edit.png")
        print("image saved at", os.path.abspath("output_image_edit.png"))