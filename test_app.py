import unittest

from app import convert_specimens


class PathologyFormatterRegressionTests(unittest.TestCase):
    def assert_formatted(self, specimen_input, expected_output):
        actual = convert_specimens(specimen_input)
        self.assertEqual(actual, expected_output)

    def test_delphian_level_variant(self):
        self.assert_formatted(
            "A. Delphian lymph node level 4",
            "FINAL DIAGNOSIS\n"
            "A. Lymph node, level IV Delphian, excision (fs):\n"
            "-",
        )

    def test_right_central_neck_level_variant(self):
        self.assert_formatted(
            "B. Right central neck level 3",
            "FINAL DIAGNOSIS\n"
            "B. Lymph node, right central neck level III, dissection:\n"
            "-",
        )

    def test_right_central_neck_nerve_insertion_variant(self):
        self.assert_formatted(
            "E. Right central neck lymph nodes at nerve insertion",
            "FINAL DIAGNOSIS\n"
            "E. Lymph nodes, right central neck at nerve insertion, dissection:\n"
            "-",
        )

    def test_right_lateral_neck_anterior_v_variant(self):
        self.assert_formatted(
            "F. Right lateral neck dissection, level 2, 3, 4 and anterior 5",
            "FINAL DIAGNOSIS\n"
            "F. Lymph nodes, right neck, levels II, III, IV, and anterior V, dissection:\n"
            "-",
        )

    def test_right_thyroid_lobe_and_isthmus_variant(self):
        self.assert_formatted(
            "B. Right thyroid lobe and isthmus",
            "FINAL DIAGNOSIS\n"
            "B. Thyroid gland, right lobe and isthmus, thyroidectomy (including fs):\n"
            "-",
        )

    def test_right_superior_parathyroid_biopsy_variant(self):
        self.assert_formatted(
            "C. Right superior parathyroid biopsy",
            "FINAL DIAGNOSIS\n"
            "C. Parathyroid, right superior, biopsy (fs):\n"
            "-",
        )

    def test_middle_meatus_laterality_variant(self):
        self.assert_formatted(
            "A. Left middle meatus mass",
            "FINAL DIAGNOSIS\n"
            "A. Nasal cavity, left middle meatus, excision (fs):\n"
            "-",
        )

    def test_lip_border_of_skin_variant(self):
        self.assert_formatted(
            "B. Right lower lip superior lateral border of skin",
            "FINAL DIAGNOSIS\n"
            "B. Lip, right lower, superior lateral border of skin, excision:\n"
            "-",
        )

    def test_site_specific_margin_preserves_margin_wording(self):
        self.assert_formatted(
            "A. Right inferior turbinate margin",
            "FINAL DIAGNOSIS\n"
            "A. Nasal cavity, right inferior turbinate margin, excision (fs):\n"
            "-",
        )

    def test_ambiguous_margin_uses_context(self):
        self.assert_formatted(
            "A. Right base of tongue\nB. Revised inferior and lateral margin",
            "FINAL DIAGNOSIS\n"
            "A. Oropharynx, right base of tongue, biopsy (fs):\n"
            "-\n\n"
            "B. Oropharynx, right base of tongue, revised inferior and lateral margin, excision (fs):\n"
            "-",
        )


if __name__ == "__main__":
    unittest.main()
