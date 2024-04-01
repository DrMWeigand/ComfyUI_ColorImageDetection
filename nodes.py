import numpy as np
import torch
import cv2
import comfy.model_management
from scipy.stats import entropy
from scipy.stats import gaussian_kde

class ColorDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", ),
            "threshold": ("FLOAT", {"default": 15.0}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("color_status", "kl_divergence")
    FUNCTION = "process"

    CATEGORY = "Image Analysis"

    @torch.no_grad()
    def process(self, image, threshold):
        self.device = comfy.model_management.get_torch_device()
        batch_size = image.shape[0]

        out = []
        for i in range(batch_size):
            img = image[i].numpy().astype(np.float32)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            deviations = []

            # Calculate the mean deviation from the mean color value per pixel
            mean_color = np.mean(img_rgb, axis=2, keepdims=True)
            deviation = np.abs(img_rgb - mean_color)
            mean_deviation = np.mean(deviation)

            # Create two-color combinations
            combos = [(img_rgb[:, :, 0], img_rgb[:, :, 1]),
                      (img_rgb[:, :, 0], img_rgb[:, :, 2]),
                      (img_rgb[:, :, 1], img_rgb[:, :, 2])]


            # Now, for the combos, calculate their mean deviations directly
            combo_deviations = []
            for combo in combos:
                combo_mean = np.mean(np.stack(combo), axis=0)
                combo_deviation = np.abs(combo[0] - combo_mean) + np.abs(combo[1] - combo_mean)
                combo_deviations.append(np.mean(combo_deviation))  # Calculate mean deviation for each combo

            # Then, calculate the overall mean deviation including the initial deviation and the combo deviations
            overall_mean_deviation = np.min([mean_deviation] + combo_deviations)
            
            # Calculate the overall mean deviation
            #mean_deviation = np.mean(overall_mean_deviation)                                
            is_color = np.mean(mean_deviation) > threshold
            out.append(("Color" if is_color else "Black and White", deviation))
    
        return (out,)


NODE_CLASS_MAPPINGS = {
    "ColorDetection": ColorDetection,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorDetection": "Color Detection",
}
