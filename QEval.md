# Quality Evaluation

## Visual realism
Evaluate the visual realism across different dimensions:

- **LPIPS**: Measure the similarity between the real image and its augmented version. 
- **SSIM**: Measure the average score in real images and synthetic images, to evaluate that fine details and textures are maintained.
- **OLIP**:  Use OLIP model to different aspects (texture, shape, joint realism)

## Label accuracy
- **Quality of generated text**: Use an OCR model to extract the text from the image, then compare the extracted text with the ground truth. Since we are estimating the theoretical upper bound of the text readability, it is important to use a large, state-of-the-art model for this task.

- **Quality of inserted objects**: Use a grounding model to verify that the inserted object can be retrieved with the right class from the image, and measure the average retrieval in the dataset.
An additional step is to retrieve the edited zone by using a segmentation model to segment the real and augmented image, and subtract them to find an editing map. Comparing the intersection of the editing map and the output of the grounding model can provide an indication of the correctess of the edits.

## Diversity of generated samples
- **FID**: Measure the realism and diversity by using the average FID score of the real images and the synthetic images in the dataset.

## Usefulness for training
Train the same architecture on the dataset comprising real and augmented images, changing the rate of real and synthetic images in the dataset. 
- Same dataset size: evaluate the performance difference between training on real images, synthetic images, or a mix of them. It may be expected that the real dataset lead to a better performance.
- Different dataset size: evaluate the performance gain on having more (synthetic) images, with respect on training on real images only, keeping the number of training steps constant. 