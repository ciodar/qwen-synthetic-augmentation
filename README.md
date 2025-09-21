# Model Card

## Method

The model uses a Deep Generative Model for image editing (Qwen or Kontext). Both models support both text-based and image-based editing. The models are loaded using the `load_qwen_image_edit` and `load_flux_kontext` functions, respectively. The models are then used to edit a source image, based on the input text or image.

## Inputs / Outputs

### Inputs
- **Text**: The text to be added to the image.
- **Image**: The image to be edited.
- **Overlay Image**: The image to be inserted into the base image.

### Outputs
- **Edited Image**: The image after the text or overlay image has been added.

## Environment

### Dependencies
- Python 3.8 or higher
- PyTorch 2.0 or higher
- Diffusers 0.26.0 or higher
- Transformers 4.36.0 or higher
- Hugging Face Hub
- PIL
- Nunchaku

### Memory Usage
- The model requires a significant amount of memory, especially when loading the Qwen and Kontext models.

### Average Processing Time
- The processing time varies depending on the complexity of the image and the text or overlay image. However, it typically takes a few seconds to a minute to edit an image.

## Limitations

### Challenges with Realism
- The model may struggle to generate realistic images, especially when the text or overlay image is complex.

### Edge Cases
- The model may not handle edge cases well, such as when the text or overlay image is too large or too small.

### Artifacts
- The model may produce artifacts in the edited image, especially when the text or overlay image is not well-aligned.

## Improvement Ideas

### Suggestions to Improve Realism
- Use a larger and more complex model to improve the realism of the edited images.

### Automation
- Automate the process of editing images by integrating the model with a user interface or a web application.

### Reusability
- The model can be adapted to other domains, such as video editing or 3D modeling, by modifying the input and output formats.

## Reusability

The model is highly adaptable to other domains, such as video editing or 3D modeling, by modifying the input and output formats. The model can also be integrated with other models or tools to enhance its functionality.