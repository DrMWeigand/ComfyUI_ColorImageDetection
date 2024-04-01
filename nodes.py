import numpy as np
import torch
import cv2
import comfy.model_management

# ColorDetection: Detects if an image is colored or black and white using RGB color space.
# It calculates mean deviation from the pixel color mean, considering a percentage of highest deviations.
# Inputs:
# - image: The image tensor to analyze.
# - threshold: The deviation threshold for determining color presence.
# - det_pixel_percent: Percentage of pixels to consider for the highest deviations analysis.
# Returns:
# - is_color (BOOL): True if the image is considered colored, False if black and white.
# - mean_deviation (FLOAT): The mean of the calculated deviations.
class ColorDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", ),
            "threshold": ("FLOAT", {"default": 0.15}), # Threshold for b&w detection adjusted based on empirical observation
            "det_pixel_percent": ("FLOAT", {"default": 0.1}),  # Percentage of pixels as a new parameter
            },
        }

    RETURN_TYPES = ("BOOL", "FLOAT")
    RETURN_NAMES = ("is_color", "mean_deviation")
    FUNCTION = "process"

    CATEGORY = "Image Analysis"

    @torch.no_grad()
    def process(self, image, threshold, det_pixel_percent):
        self.device = comfy.model_management.get_torch_device()
        batch_size = image.shape[0]

        out = []
        for i in range(batch_size):
            img = image[i].numpy().astype(np.float32)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            deviations = np.abs(img_rgb - np.mean(img_rgb, axis=2, keepdims=True)).flatten()
            
            # Use the provided percentage of pixels for deviation calculation
            num_pixels_to_consider = int(len(deviations) * (det_pixel_percent / 100.0))
            mean_deviation = np.mean(np.sort(deviations)[-num_pixels_to_consider:])

            is_color = mean_deviation > threshold
            #out.append(("Color" if is_color else "Black and White", mean_deviation))
            out = "Color" if is_color else "Black and White"
            
        return (is_color, mean_deviation)


# LABColorDetection: Differentiates colored images from black and white ones using the LAB color space.
# It assesses color presence by analyzing differences between the A and B channels.
# Inputs:
# - image: The image tensor to analyze.
# - threshold: The difference threshold between A and B channels to consider the image colored.
# Returns:
# - is_color (BOOL): True if the image is considered colored, False if black and white.
# - color_difference (FLOAT): The mean difference between A and B channels. 
class LABColorDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", ),
            "threshold": ("FLOAT", {"default": 2.5}),  # Threshold adjusted based on empirical observation
            },
        }

    RETURN_TYPES = ("BOOL", "FLOAT")
    RETURN_NAMES = ("is_color", "color_difference")
    FUNCTION = "process"

    CATEGORY = "Image Analysis"

    @torch.no_grad()
    def process(self, image, threshold):
        self.device = comfy.model_management.get_torch_device()
        batch_size = image.shape[0]

        out = []
        for i in range(batch_size):
            img = image[i].numpy().astype(np.float32)
            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab_img)
            color_difference = np.mean(np.abs(a_channel - b_channel))

            is_color = color_difference > threshold

        return (is_color, color_difference)

NODE_CLASS_MAPPINGS = {
    "RGBColorDetection": ColorDetection,
    "LABColorDetection": LABColorDetection,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorDetection": "RGB Color Detection",
    "LABColorDetection": "LAB Color Detection",
}