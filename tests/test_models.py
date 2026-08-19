"""
Unit tests for neural model interfaces and contract definitions.
"""

import unittest
from src.models.time_aware_encoder import TimeAwareEncoder
from src.models.latent_projection import LatentProjection
from src.models.neural_sde import LatentNeuralSDE
from src.models.decoder import ObservationDecoder


class TestModelInterfaces(unittest.TestCase):

    def test_model_interface_instantiation(self):
        """Verify that PyTorch model modules can be instantiated with dimension configs."""
        encoder = TimeAwareEncoder(input_dim=10, hidden_dim=64)
        proj = LatentProjection(hidden_dim=64, latent_dim=32)
        sde = LatentNeuralSDE(latent_dim=32, hidden_dim=64)
        decoder = ObservationDecoder(latent_dim=32, output_dim=10, hidden_dim=64)

        self.assertEqual(encoder.input_dim, 10)
        self.assertEqual(encoder.hidden_dim, 64)
        self.assertEqual(proj.latent_dim, 32)
        self.assertEqual(sde.latent_dim, 32)
        self.assertEqual(decoder.output_dim, 10)


if __name__ == "__main__":
    unittest.main()
