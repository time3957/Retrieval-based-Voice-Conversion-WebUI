import json
import locale
import os
from pathlib import Path

from tools.file_io import read_text


LOCALE_DIR = Path(__file__).parent / "locale"
DEFAULT_LANGUAGE_FILE = Path(__file__).parent / "selected_language.txt"


def load_language_list(language):
    return json.loads(read_text(str(LOCALE_DIR / f"{language}.json")))


def get_preferred_language(settings_path=DEFAULT_LANGUAGE_FILE):
    """Return the saved language, or None when the user chose automatic mode."""
    path = Path(settings_path)
    if not path.is_file():
        return None

    language = path.read_text(encoding="utf-8").strip()
    supported = {file.stem for file in LOCALE_DIR.glob("*.json")}
    return language if language in supported else None


def save_preferred_language(language, settings_path=DEFAULT_LANGUAGE_FILE):
    """Save a supported language; Auto removes the override and uses the OS locale."""
    path = Path(settings_path)
    if language == "Auto":
        path.unlink(missing_ok=True)
        return

    supported = {file.stem for file in LOCALE_DIR.glob("*.json")}
    if language not in supported:
        raise ValueError(f"Unsupported language: {language}")

    path.write_text(language, encoding="utf-8")


class I18nAuto:
    def __init__(self, language=None, settings_path=DEFAULT_LANGUAGE_FILE):
        if language in ["Auto", None]:
            language = get_preferred_language(settings_path)
            if language is None:
                language = locale.getdefaultlocale()[0]
        if not os.path.exists(LOCALE_DIR / f"{language}.json"):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language
