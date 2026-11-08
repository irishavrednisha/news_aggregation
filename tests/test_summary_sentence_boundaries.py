"""Test TextRank sentence boundaries without loading ML or DB dependencies."""
import ast
from pathlib import Path
import re
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "processing" / "summarizer.py"


def load_split_into_sentences():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "split_into_sentences"]
    if len(matches) != 1:
        raise AssertionError("split_into_sentences not found")
    namespace = {
        "re": re,
        "normalize_article_text": lambda text: text,
        "BAD_SUMMARY_PHRASES": [],
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    return namespace["split_into_sentences"]


class SummarySentenceTests(unittest.TestCase):
    def test_580_thousand_is_kept_with_unit(self):
        split = load_split_into_sentences()
        text = (
            "Предприятие расширило производство гильз цилиндров "
            "до 580 тыс. изделий в год. "
            "Другой цех продолжает выпуск оборудования для предприятий."
        )
        result = split(text)
        self.assertEqual(len(result), 2, result)
        self.assertIn("580 тыс. изделий в год.", result[0])

    def test_square_meter_abbreviation(self):
        split = load_split_into_sentences()
        text = (
            "За последние шесть лет застройщики ввели "
            "67 млн кв. метров жилья в различных регионах. "
            "Программа продолжает действовать в нескольких субъектах."
        )
        result = split(text)
        self.assertEqual(len(result), 2, result)
        self.assertIn("67 млн кв. метров жилья", result[0])


if __name__ == "__main__":
    unittest.main()
