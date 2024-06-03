import os
import torch
import numpy as np
from PIL import Image
from Models.Generators import ConditionalVAE

class HandwrittenTextGenerator:
    """Class to generate handwritten text using a Conditional VAE."""

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

    def create_space_image(self, width=28, height=56):
        """Create a blank image."""
        return np.zeros((height, width))

    def convert_word(self, word, current_line, current_length, add_space=True):
        for char in word:
            img = self.generate_character_image(char)
            img = np.kron(img, np.ones((2, 2)))  # Scale up the image
            img = (img * 255).astype(np.uint8)
            current_line.append(img)

        # Update the length of the line after the word is added
        current_length += len(word)
        if add_space:
            img = self.create_space_image()  # add blank space
            current_line.append(img)
            current_length += 1  # increase length by 1 to include the space
        return current_line, current_length

    def generate_handwritten_images(self, text, max_chars_per_line=25):
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            # Check if the word can be added to the line or not
            if current_length + len(word) <= max_chars_per_line:
                # Check if we need to add space at the end of the word or if it is the end of the line
                add_space = current_length + len(word) != max_chars_per_line
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

# # Example usage
# if __name__ == "__main__":
#     generator = HandwrittenTextGenerator()
#     text = "Hello World"
#     images = generator.generate_handwritten_images(text)
#     for img in images:
#         print(img)
