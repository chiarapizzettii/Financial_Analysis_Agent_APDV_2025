from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0


def get_user_language(text):
    if not text or len(text.strip()) < 3:
        return "English"

    try:
        # Restituisce codici come 'it', 'en', 'fr'
        lang_code = detect(text)

        # Mappatura per rendere il risultato leggibile
        languages = {
            "it": "Italiano",
            "en": "English",
            "fr": "Français",
            "es": "Español",
        }

        return languages.get(lang_code, "English")  # Default English

    except Exception:
        return "English"
