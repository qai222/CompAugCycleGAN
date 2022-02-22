import unittest


class LayerTest(unittest.TestCase):

    def test_filter(self):
        from cacgan.gans.networks import ConditionalResnetGenerator, torch
        from cacgan.utils import DEVICE
        input_size = 6
        ninput = 3
        zero_columns = [2, 3]
        generator_input = torch.rand((ninput, input_size), device=DEVICE)
        for icol in zero_columns:
            generator_input[:, icol] = 0
        crg = ConditionalResnetGenerator(6, input_size, input_size)
        crg.model.to(device=DEVICE)
        noise = torch.rand((ninput, input_size), device=DEVICE)
        out = crg.forward(generator_input, noise)
        for icol in zero_columns:
            self.assertTrue(all(out[:, icol] == 0))


if __name__ == '__main__':
    unittest.main()
