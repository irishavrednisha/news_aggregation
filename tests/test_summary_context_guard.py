"""News summary selection must reject dangling context, not invent referents.

The tests execute real production functions without importing DB/ML models.
"""
import ast
from pathlib import Path
import re
import unittest

PATH = Path(__file__).resolve().parents[1] / "processing" / "summarizer.py"


def functions_for_context_tests():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    names = {
        "normalize_sentence_key", "starts_badly", "is_bad_summary_sentence",
        "build_textrank_summary",
    }
    found = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(found) != len(names):
        raise AssertionError("Expected all production summary functions")
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                    "BAD_STARTS", "BAD_CONTEXT_PHRASES"}:
                constants[target.id] = ast.literal_eval(node.value)
    ns = {
        **constants, "re": re,
        "split_into_sentences": lambda text: [x.strip() for x in
                                             re.split(r"(?<=[.!?])\s+", text)],
        "remove_title_like_sentences": lambda x, title: x,
        "remove_duplicate_sentences": lambda x: x,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=found, type_ignores=[])),
                 str(PATH), "exec"), ns)
    return ns


class SummaryContextGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = functions_for_context_tests()

    def test_avoid_contextless_quotes_from_housing_article(self):
        bad = (
            "Мы буквально вчера завершили распределение субсидий по ней на субъекты РФ.",
            "И это теперь создает сложности в плане закрытия проекта КРТ.",
            "А на следующий год компания получила статус резидента технополиса.",
            "Отметим, что в сварочном производстве площадки действуют 175 роботов.",
        )
        for sentence in bad:
            with self.subTest(sentence=sentence):
                self.assertTrue(self.ns["is_bad_summary_sentence"](sentence))

    def test_independent_industrial_event_is_not_removed(self):
        good = (
            "Российское предприятие разработало температурный датчик для химических реакторов.",
            "Онлайн-производство комплектующих запустило новую автоматизированную линию.",
            "Этап модернизации производственных мощностей завершился в конце года.",
        )
        for sentence in good:
            with self.subTest(sentence=sentence):
                self.assertFalse(self.ns["is_bad_summary_sentence"](sentence))

    def test_no_fallback_to_incoherent_extract(self):
        source = (
            "Мы буквально вчера завершили распределение субсидий по ней на субъекты РФ. "
            "И это теперь создает сложности в плане закрытия проекта КРТ."
        )
        self.assertEqual(self.ns["build_textrank_summary"](source), "")


if __name__ == "__main__":
    unittest.main()
