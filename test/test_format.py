"""`{...}` is a format field, `<<...>>` is a formula str.format must not touch."""
import unittest

PARAMS = {'accuracy': 0.9, 'duration': 0.1, 'model_name': 'test'}


class TestPromptFormatting(unittest.TestCase):
    def test_angle_delimited_formulas_survive_format(self):
        text = "Result: <<accuracy / duration>> ns⁻¹".format(**PARAMS)
        self.assertEqual("Result: <<accuracy / duration>> ns⁻¹", text)

    def test_doubled_braces_become_literal_braces(self):
        text = "Result: {{accuracy / duration}} ns⁻¹".format(**PARAMS)
        self.assertEqual("Result: {accuracy / duration} ns⁻¹", text)

    def test_a_single_braced_name_is_substituted(self):
        self.assertEqual("acc 0.9", "acc {accuracy}".format(**PARAMS))


if __name__ == '__main__':
    unittest.main()
