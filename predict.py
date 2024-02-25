from cog import BasePredictor, Input, Path
import torch
import tempfile
from oot_diffusion import OOTDiffusionModel


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = OOTDiffusionModel()
        self.model.load_pipe()

        return model

    # The arguments and types the model takes as input
    def predict(
        self,
        model_image: Path = Input(description="Grayscale input image"),
        garment_image: Path = Input(description="Grayscale input image"),
        steps: int = Input(
            default=20, description="Grayscale input image", min=1, max=40
        ),
        guidance_scale: float = Input(
            default=2.0, description="Guidance scale", min=1.0, max=5.0
        ),
        seed: int = Input(
            default=0, description="Grayscale input image", min=0, max=1000000
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        generated_images, mask_image = self.model.generate(
            model_path=model_image,
            cloth_image=garment_image,
            steps=steps,
            cfg=guidance_scale,
            seed=seed,
            num_samples=1,
        )

        result_path = Path(tempfile.mktemp(suffix=".jpg"))

        generated_images[0].save(result_path, "JPEG")

        return result_path
