import os
import sys
import unittest
import json

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever_server.vector_retriever import VectorRetriever
from retriever_server.unified_retriever import UnifiedRetriever
from commaqa.models.mock_generator import MockGenerator
from evaluate import evaluate_by_dicts

class TestIRCoTPipeline(unittest.TestCase):

    def setUp(self):
        self.raw_data_dir = "raw_data"
        self.processed_data_dir = "processed_data"

    def test_vector_retriever(self):
        retriever = VectorRetriever()
        results = retriever.retrieve_paragraphs(
            corpus_name="hotpotqa",
            query_text="Scott Derrickson",
            max_hits_count=5
        )
        self.assertGreater(len(results), 0)
        self.assertIn("Scott Derrickson", results[0]["title"])
        self.assertIn("corpus_name", results[0])

    def test_unified_retriever_fallback(self):
        unified = UnifiedRetriever()
        results = unified.retrieve_from_elasticsearch(
            query_text="Arthur's Magazine",
            corpus_name="hotpotqa",
            max_hits_count=3
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["title"], "Arthur's Magazine")

    def test_mock_generator_reasoning(self):
        gen = MockGenerator()
        res = gen.generate_text_sequence("Were Scott Derrickson and Ed Wood of the same nationality?")
        self.assertGreater(len(res), 0)
        self.assertIn("yes", res[0][0].lower())

    def test_evaluation_metrics(self):
        predictions = {
            "hp_sample_1": "yes",
            "hp_sample_2": "Arthur's Magazine"
        }
        id_to_ground_truths = {
            "hp_sample_1": (["yes"], ["Scott Derrickson"]),
            "hp_sample_2": (["Arthur's Magazine"], ["Arthur's Magazine"])
        }
        # evaluate_by_dicts expects id_to_ground_truths maps id -> (answer_list, supporting_titles)
        # where answer_list is [ "yes" ] or [ ["yes"] ]
        id_to_ground_truths_ans = {
            "hp_sample_1": ["yes"],
            "hp_sample_2": ["Arthur's Magazine"]
        }
        metrics = evaluate_by_dicts("answer", id_to_ground_truths_ans, predictions)
        self.assertEqual(metrics["em"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)





if __name__ == "__main__":
    unittest.main()
