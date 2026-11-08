"""Regression: contact cleanup must not eat the prefix of 'телевизор'."""
import unittest

from utils.text_utils import normalize_article_text, remove_contacts_and_service_text


class ContactCleanupTests(unittest.TestCase):
    def test_smart_televisions_are_preserved(self):
        text = (
            "На бывшем заводе Mazda во Владивостоке компания "
            "«Альтер Инвест» начала выпуск умных телевизоров."
        )
        actual = normalize_article_text(text)
        self.assertIn("телевизоров", actual)
        self.assertNotIn("умных евизоров", actual)

    def test_phone_label_still_removed(self):
        cleaned = remove_contacts_and_service_text(
            "Связаться: тел. Телевизоры производятся на заводе."
        )
        self.assertNotIn("тел.", cleaned.casefold())
        self.assertIn("Телевизоры", cleaned)


if __name__ == "__main__":
    unittest.main()
