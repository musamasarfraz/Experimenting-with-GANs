import string
import random
import torch
import numpy as np
from PIL import Image
from Models.Generators import ConditionalGeneratorUppercase, ConditionalGeneratorLowercase, ConditionalGeneratorDigits

class CaptchaGenerator:
    """Class to generate CAPTCHA images and text."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator_uppercase = None
        self.generator_lowercase = None
        self.generator_digits = None
        self.initialize_generators()

    def initialize_generators(self):
        """Initialize conditional generators."""
        self.generator_uppercase = self.load_model(ConditionalGeneratorUppercase, 'Models/GAN_generator_uppercase.pth')
        self.generator_lowercase = self.load_model(ConditionalGeneratorLowercase, 'Models/GAN_generator_lowercase.pth')
        self.generator_digits = self.load_model(ConditionalGeneratorDigits, 'Models/GAN_generator_digits.pth')

    def load_model(self, model_class, checkpoint_path):
        """Load a generator model from a checkpoint."""
        model = model_class().to(self.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        return model

    def generate_character_image(self, generator, char, char_map, is_letter=False):
        """Generate an image for a given character."""
        char_idx = torch.tensor([char_map[char]]).to(self.device)
        z = torch.randn(1, 100).to(self.device)
        with torch.no_grad():
            gen_img = generator(z, char_idx).cpu().numpy()
        gen_img = gen_img.squeeze()
        gen_img = (gen_img + 1) / 2.0  # Normalize to [0, 1]
        if is_letter:
            gen_img = self.correct_orientation(gen_img)
        return gen_img

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
        digit_map = {str(i): i for i in range(10)}
        letter_map_upper = {chr(i): i - 65 for i in range(65, 91)}  # A-Z
        letter_map_lower = {chr(i): i - 97 for i in range(97, 123)}  # a-z

        images = []
        for char in text:
            if char.isdigit():
                img = self.generate_character_image(self.generator_digits, char, digit_map)
            elif char.isupper():
                img = self.generate_character_image(self.generator_uppercase, char, letter_map_upper, is_letter=True)
            elif char.islower():
                img = self.generate_character_image(self.generator_lowercase, char, letter_map_lower, is_letter=True)
            else:
                img = self.create_space_image()  # Handles spaces or unrecognized characters
            # Resize image to increase dimensions
            img = np.kron(img, np.ones((2, 2)))  # Scale up the image by 2x for example
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
