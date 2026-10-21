import json
import re
import tempfile
import unittest
from pathlib import Path

from i18n.i18n import I18nAuto, get_preferred_language, save_preferred_language


ROOT = Path(__file__).resolve().parents[1]


class ThaiLocaleTests(unittest.TestCase):
    def test_thai_locale_has_the_same_keys_as_english(self):
        english = json.loads((ROOT / "i18n/locale/en_US.json").read_text(encoding="utf-8"))
        thai = json.loads((ROOT / "i18n/locale/th_TH.json").read_text(encoding="utf-8"))
        self.assertEqual(set(english), set(thai))
        self.assertEqual(thai["模型推理"], "แปลงเสียง")

    def test_format_placeholders_match_english(self):
        english = json.loads((ROOT / "i18n/locale/en_US.json").read_text(encoding="utf-8"))
        thai = json.loads((ROOT / "i18n/locale/th_TH.json").read_text(encoding="utf-8"))
        pattern = re.compile(r"%[-+#0 ]*(?:\d+)?(?:\.\d+)?[diouxXeEfFgGcrsa%]|\{[^{}]*\}")
        for key, english_text in english.items():
            self.assertEqual(
                pattern.findall(english_text),
                pattern.findall(thai[key]),
                f"Placeholder mismatch for {key}",
            )

    def test_saved_language_is_used_on_next_start(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "selected_language.txt"
            save_preferred_language("th_TH", settings_file)
            self.assertEqual(get_preferred_language(settings_file), "th_TH")
            self.assertEqual(I18nAuto(settings_path=settings_file).language, "th_TH")

    def test_invalid_language_is_not_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "selected_language.txt"
            with self.assertRaises(ValueError):
                save_preferred_language("not_a_language", settings_file)
            self.assertFalse(settings_file.exists())

    def test_thai_can_be_loaded_explicitly(self):
        thai = I18nAuto("th_TH")
        self.assertEqual(thai("模型推理"), "แปลงเสียง")


if __name__ == "__main__":
    unittest.main()
