# Model Card

## Method

The model uses a Deep Generative Model for image editing (Qwen or Kontext). Both models support both text-based and image-based editing. The models are loaded using the `load_qwen_image_edit` and `load_flux_kontext` functions, respectively. The models are then used to edit a source image, based on the input text or image.

## Inputs / Outputs
### Script Inputs
- **csv_path** (str): Path to the CSV file containing the input data to be augmented.
- **output_dir** (str): Directory where the augmented images will be saved.

### CSV Fields
- **image_id** (str): COCO image ID of the input image. 
- **overla_text** (str, optional): File path to the text file containing the text to be added.
- **overlay_image** (str, optional): File path to the image to overlay. Leave blank if using `overlay_text`.
- **prompt** (str): Text prompt to be used to guide the image editing for the line

### Outputs

- **Edited Image**: The image after the text or overlay image has been added.

## Environment

### Dependencies
- Python 3.10 or higher
- diffusers 0.36.0 nightly
- transformers 4.56.3 nightly
- torch 2.8.0
- gguf >=0.10.0
- accelerate

### Inference
This script requires a significant amount of memory due to the usage of Qwen or Flux Kontext models.

The pipeline was tested on a A100 GPU with 40GB of VRAM, with an average of 24 GB of vRAM and employed 160s for each image.

### Reduce memory usage/inference time
The pipeline can be made significantly faster and less memory intensive by 
- Employing [offloading techniques](https://huggingface.co/docs/diffusers/en/optimization/memory#offloading)
- Using more heavily quantized models (See https://huggingface.co/QuantStack/Qwen-Image-GGUF)
- Use [nunchaku](https://github.com/nunchaku-tech/nunchaku)
- Use step distillation loras such as [LightX2v](https://huggingface.co/lightx2v/Qwen-Image-Lightning)

## Limitations

- **Processing time**: This technique is computing intensive and requires a significant amount of time to make inference, even on high-end GPU. Using the steps described in the above section can help mitigate the issue.

- **Prompt adherence**: The model may not adhere to the input prompt exactly, and the input image can be distorted or 

- **Artifacts**: The model may produce artifacts in the edited image, or alter details both in the source image and the overlay image. Using a mask for inpainting can help to mitigate the alterations on the source image.

- **Subject size**: This pipeline is designed for subjects that are centered in the image, and occupying a significant portion of the image. Smaller subjects can result in no augmentaton, or worse quality.

## Improvement Ideas
- **Avoid prompt**: right now the augmentation process takes a prompt as input. This could be avoided by exploiting the dataset categories. To improve qualitative results and the quality of the intended edit, the pipeline could use the dataset's textual captions (if available), or employ an Image Captioning model to describe the input image and the overlay. 

## Reusability
This pipeline uses a general image editing model and can be adapted to most use cases involving everyday life scenes. Model adaptations or dedicated model use or fine-tuning could be required for specific domains (Medical, manifacturing, ...)