import string
import random
import torch
import numpy as np
from PIL import Image
from Models.Generators import ConditionalVAE

class CaptchaGenerator:
    """Class to generate CAPTCHA images and text using a Conditional VAE."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae_model = self.load_model(ConditionalVAE, 'Models/conditional_vae_emnist_epoch280.pth')
        self.char_mapping = self.create_char_mapping()

    def load_model(self, model_class, checkpoint_path):
        """Load a VAE model from a checkpoint."""
        model = model_class().to(self.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        return model

    def create_char_mapping(self):
        """Create character to index mapping for digits, uppercase, and lowercase letters."""
        char_mapping = {chr(i + 48): i for i in range(10)}  # {'0': 0, '1': 1, ..., '9': 9}
        char_mapping.update({chr(i + 65): i + 10 for i in range(26)})  # {'A': 10, 'B': 11, ..., 'Z': 35}
        char_mapping.update({chr(i + 97): i + 36 for i in range(26)})  # {'a': 36, 'b': 37, ..., 'z': 61}
        char_mapping[' '] = "space"  # Special entry for spaces
        return char_mapping

    def generate_character_image(self, char):
        """Generate an image for a given character using the VAE."""
        if char == ' ':
            return self.create_space_image()
        else:
            char_index = self.char_mapping[char]
            return self.generate_character(self.vae_model, char_index)

    def generate_character(self, vae_model, char_index):
        """Generate a character image using the VAE model."""
        vae_model.eval()
        with torch.no_grad():
            sample_z = torch.randn(1, 32).to(self.device)  # Assuming the latent dimension size is 32
            labels = torch.tensor([char_index], dtype=torch.long).to(self.device)
            generated_image = vae_model.decode(sample_z, labels).cpu()
            image = generated_image.squeeze(0).view(28, 28).numpy()  # Reshape to the image dimensions
            return self.correct_orientation(image)

    def correct_orientation(self, image):
        """Correct the orientation of an image."""
        rotated_image = np.rot90(image, k=3)
        corrected_image = np.fliplr(rotated_image)
        return corrected_image

    def create_space_image(self, width=28, height=28):
        """Create a blank image."""
        return np.ones((height, width)) * -1

    def generate_captcha_text(self, length=5):
        """Generate random CAPTCHA text."""
        chars = string.ascii_letters + string.digits  # Include both uppercase and lowercase
        return ''.join(random.choice(chars) for _ in range(length))

    def create_captcha_image(self, text):
        """Create a CAPTCHA image for the given text."""
        images = []
        for char in text:
            img = self.generate_character_image(char)
            img = np.kron(img, np.ones((2, 2)))  # Scale up the image by 2x
            images.append(img)

        total_width = sum(img.shape[1] for img in images)
        max_height = max(img.shape[0] for img in images)
        combined_image = np.zeros((max_height, total_width))

        current_x = 0
        for img in images:
            combined_image[:img.shape[0], current_x:current_x + img.shape[1]] = img
            current_x += img.shape[1]

        combined_image = (combined_image * 255).astype(np.uint8)
        img_pil = Image.fromarray(combined_image)
        img_pil.save('static/captcha.png')