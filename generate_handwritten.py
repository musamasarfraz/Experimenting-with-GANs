import os
import torch
import numpy as np
from PIL import Image
from Models.Generators import ConditionalGeneratorUppercase, ConditionalGeneratorLowercase, ConditionalGeneratorDigits

class HandwrittenTextGenerator:
    """Class to generate handwritten text using GAN generators."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator_uppercase = None
        self.generator_lowercase = None
        self.generator_digits = None
        self.initialize_generators()
        self.letter_map_upper = {chr(i): i - 65 for i in range(65, 91)}  # A-Z
        self.letter_map_lower = {chr(i): i - 97 for i in range(97, 123)}  # a-z
        self.digit_map = {str(i): i for i in range(10)}

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
        if char not in char_map:
            print(f"Character '{char}' not found in the map, generating space instead.")
            return self.create_space_image(width=28, height=28)  # Return a space image if character not found
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

    def create_space_image(self, width=28, height=56):
        """Create a blank image."""
        return np.zeros((height, width))

    def convert_word(self, word, current_line, current_length, add_space = True):
        for char in word:
            if char.isdigit():
                img = self.generate_character_image(self.generator_digits, char, self.digit_map)
            elif char.isupper():
                img = self.generate_character_image(self.generator_uppercase, char, self.letter_map_upper, is_letter=True)
            elif char.islower():
                img = self.generate_character_image(self.generator_lowercase, char, self.letter_map_lower, is_letter=True)
            img = np.kron(img, np.ones((2, 2)))  # Scale up the image
            img = (img * 255).astype(np.uint8)
            current_line.append(img)
            
        # Update the length of the line after the word is added
        current_length += len(word)
        if add_space:
            img = self.create_space_image() # add blank space
            current_line.append(img)
            current_length += 1 # increase length by 1 to include the space
        return current_line, current_length


    def generate_handwritten_images(self, text, max_chars_per_line=25):
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # Check if the word can be added to the line or not
            if current_length + len(word) <= max_chars_per_line:
                # Check if we need to add space at the end of the or nor is it end of the line
                if current_length + len(word) == max_chars_per_line:
                    add_space = False
                else:
                    add_space = True
                # Add the word to the current line
                current_line, current_length = self.convert_word(word, current_line, current_length, add_space)
                
            else:
                # if there is not enough space to add the word we fill the remaining length with spaces
                padding = max_chars_per_line - current_length
                for pad in range(padding):
                    current_line.append(self.create_space_image())
                lines.append(current_line)
                
                # we start the new line to include remaining text
                current_line = []
                current_length = 0
                # we transfer the current word to the start of the new line
                current_line, current_length = self.convert_word(word, current_line, current_length)

        if current_line:
            padding = max_chars_per_line - current_length
            for pad in range(padding):
                current_line.append(self.create_space_image())
            lines.append(current_line)

        combined_images = []
        for line in lines:

            # Combine images into one image
            max_height = max(img.shape[0] for img in line)
            total_width = sum(img.shape[1] for img in line)
            combined_image = np.ones((max_height, total_width), dtype=np.uint8) * 255  # Start with a white background

            current_x = 0
            for img in line:
                combined_image[:img.shape[0], current_x:current_x + img.shape[1]] = img
                current_x += img.shape[1]

            # Save and store the image path
            img_pil = Image.fromarray(combined_image)
            img_path = f'static/handwritten_{len(combined_images)}.png'
            img_pil.save(img_path)
            combined_images.append(f'/static/handwritten_{len(combined_images)}.png')

        return combined_images
