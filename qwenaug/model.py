import os
from PIL import Image
import torch

from huggingface_hub import hf_hub_download

def download_qwen_weights(directory='weights'):
    original_repo = 'Qwen/Qwen-Image-Edit'
    quantized_repo = 'QuantStack/Qwen-Image-Edit-GGUF'
    # Download vae
    os.makedirs('weights/vae',exist_ok=True)
    hf_hub_download(repo_id=original_repo, filename='vae/config.json', local_dir='weights')
    hf_hub_download(repo_id=original_repo, filename='vae/diffusion_pytorch_model.safetensors', local_dir='weights')
    os.makedirs('weights/transformer',exist_ok=True)
    hf_hub_download(repo_id=original_repo, filename='transformer/config.json', local_dir='weights')
    hf_hub_download(repo_id=quantized_repo, filename='Qwen_Image_Edit-Q8_0.gguf', local_dir='weights/transformer')
    # os.makedirs('weights/text_encoder',exist_ok=True)
    # hf_hub_download(repo_id='Qwen/Qwen2.5-VL-7B-Instruct',  local_dir='weights/transformer')

def download_kontext_weights(directory='weights'):
    original_repo = 'black-forest-labs/FLUX.1-Kontext-dev'
    quantized_repo = 'QuantStack/FLUX.1-Kontext-dev-GGUF'
    os.makedirs('weights/vae',exist_ok=True)
    hf_hub_download(repo_id=original_repo, filename='vae/config.json', local_dir='weights')
    hf_hub_download(repo_id=original_repo, filename='vae/diffusion_pytorch_model.safetensors', local_dir='weights')
    os.makedirs('weights/transformer',exist_ok=True)
    hf_hub_download(repo_id=original_repo, filename='transformer/config.json', local_dir='weights')
    hf_hub_download(repo_id=quantized_repo, filename='flux1-kontext-dev-Q8_0.gguf', local_dir='weights/transformer')
    os.makedirs('weights/text_encoder/t5', exist_ok=True)
    hf_hub_download(repo_id='city96/t5-v1_1-xxl-encoder-gguf', filename='t5-v1_1-xxl-encoder-Q8_0.gguf', local_dir='weights/text_encoder/t5')

def load_flux_kontext():
    from diffusers import FluxKontextPipeline, GGUFQuantizationConfig, FluxTransformer2DModel, FluxKontextInpaintPipeline
    from diffusers.hooks import apply_layerwise_casting, apply_group_offloading
    from transformers import T5EncoderModel
    # from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
    # from nunchaku.utils import get_precision
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    transformer = FluxTransformer2DModel.from_single_file(
        pretrained_model_link_or_path_or_dict='weights/transformer/flux1-kontext-dev-Q8_0.gguf',
        quantization_config=quantization_config,
        config="weights/transformer/config.json",
        dtype=torch.bfloat16
    )
    # transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        'city96/t5-v1_1-xxl-encoder-gguf',
        gguf_file='t5-v1_1-xxl-encoder-Q8_0.gguf',
        # quantization_config=quantization_config,
        # config="weights/text_encoder/t5/config.json",
        dtype=torch.bfloat16
    )
    # transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

    # apply_layerwise_casting(
    #     text_encoder_2,
    #     storage_dtype=torch.float8_e4m3fn,
    #     compute_dtype=torch.bfloat16,
    #     non_blocking=True,
    # )

    pipe = FluxKontextInpaintPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", 
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16, 
        # device_map="balanced"
    )
    # apply_layerwise_casting(
    #     pipe.text_encoder,
    #     storage_dtype=torch.float8_e4m3fn,
    #     compute_dtype=torch.bfloat16,
    #     non_blocking=True,
    # )
    # apply_layerwise_casting(
    #     pipe.transformer,
    #     storage_dtype=torch.float8_e4m3fn,
    #     compute_dtype=torch.bfloat16,
    #     non_blocking=True,
    # )
    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")
    # apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    # apply_group_offloading(pipe.text_encoder_2, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    # pipe.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True, record_stream=True)
    # pipe.vae.enable_group_offload(onload_device=onload_device, offload_type="leaf_level", use_stream=True, record_stream=True)
    pipe.to("cuda")
    return pipe


def load_qwen_image_edit():
    from diffusers import QwenImageEditPipeline, AutoencoderKLQwenImage, QwenImageTransformer2DModel, GGUFQuantizationConfig
    from diffusers.hooks import apply_group_offloading
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from diffusers.hooks import apply_layerwise_casting
    # The original pipeline would be simply
    # pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")

    # Load VAE
    vae = AutoencoderKLQwenImage.from_pretrained(
        'weights/vae/',
        # config='weights/vae/config.json',
        torch_dtype=torch.bfloat16
    )
    print("Loaded vae!")
    # Load quantized Transformer
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    transformer = QwenImageTransformer2DModel.from_single_file(
        pretrained_model_link_or_path_or_dict='weights/transformer/Qwen_Image_Edit-Q8_0.gguf',
        quantization_config=quantization_config,
        config="weights/transformer/config.json",
        dtype=torch.bfloat16
    )
    # transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    print("Loaded transformer!")
    # Load Text Encoder
    # GGUF still not implemented see https://github.com/huggingface/transformers/issues/40049
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        #'weights/text_encoder', 
        #quantization_config=quantization_config,
        dtype=torch.bfloat16
    )
    # apply_layerwise_casting(
    #     text_encoder,
    #     storage_dtype=torch.float8_e4m3fn,
    #     compute_dtype=torch.bfloat16,
    #     non_blocking=True,
    # )
    # Load image encoder
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("loaded processor")
    
    # vae = AutoencoderKL.from_single_file(
    #     "Qwen/Qwen-Image",
    #     dtype=torch.bfloat16
    # )

    pipeline = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        processor=processor,
        dtype=torch.bfloat16
    )

    print("pipeline loaded")

    # Apply group offloading
    # pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True, record_stream=True)
    # pipeline.vae.enable_group_offload(onload_device=onload_device, offload_type="leaf_level", use_stream=True, record_stream=True)
    # Use the apply_group_offloading method for other model components
    # apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2, use_stream=True, record_stream=True)
    
    pipeline.enable_model_cpu_offload()
    
    # pipeline.to("cuda")
    
    return pipeline

    

if __name__ == "__main__":
    # Example usage:
    pipeline = load_flux_kontext()
    image = Image.open("examples/images/input.jpg").convert("RGB")
    # image.resize(48, 64)
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
        print("pipeline loaded successfully!")
        # You can now use the 'image_pipeline' object to generate images.
        # For example:
        # image = image_pipeline("A photo of a cat wearing a hat")
        # image.save("cat_with_hat.png")
    else:
        print("Failed to load pipeline.")

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save("output_image_edit.png")
        print("image saved at", os.path.abspath("output_image_edit.png"))