import json
import tempfile
import unittest
from pathlib import Path

from src.tokenizer.artifacts import load_tokenizer, resolve_tokenizer_artifact_path, save_tokenizer
from src.tokenizer.bpe import BPETokenizer


class TokenizerArtifactTests(unittest.TestCase):
    def test_save_and_load_tokenizer_artifact(self):
        tokenizer = BPETokenizer()
        tokenizer.train("banana bandana", vocab_size=20)

        artifact_dir = "artifacts/test-tokenizers"
        artifact_name = "artifact-test.json"
        artifact_path = resolve_tokenizer_artifact_path(artifact_dir, artifact_name)

        try:
            save_tokenizer(tokenizer, artifact_dir, artifact_name)
            loaded = load_tokenizer(artifact_dir, artifact_name)

            self.assertEqual(loaded.encode("banana"), tokenizer.encode("banana"))
        finally:
            if artifact_path.exists():
                artifact_path.unlink()
            if artifact_path.parent.exists():
                artifact_path.parent.rmdir()

    def test_load_tokenizer_requires_existing_artifact(self):
        with self.assertRaises(FileNotFoundError):
            load_tokenizer("artifacts/test-tokenizers", "missing.json")



class BPETokenizerTests(unittest.TestCase):
    def test_encode_before_training_raises(self):
        tokenizer = BPETokenizer()

        with self.assertRaises(ValueError):
            tokenizer.encode("abc")

    def test_training_is_deterministic(self):
        text = "banana bandana banana"
        tokenizer_a = BPETokenizer()
        tokenizer_b = BPETokenizer()

        tokenizer_a.train(text, vocab_size=40)
        tokenizer_b.train(text, vocab_size=40)

        self.assertEqual(tokenizer_a.merges, tokenizer_b.merges)
        self.assertEqual(tokenizer_a.token_to_id, tokenizer_b.token_to_id)
        self.assertEqual(tokenizer_a.encode(text), tokenizer_b.encode(text))

    def test_ascii_round_trip(self):
        tokenizer = BPETokenizer()
        tokenizer.train("hello hello", vocab_size=20)

        encoded = tokenizer.encode("hello")

        self.assertEqual(tokenizer.decode(encoded), "hello")

    def test_unicode_round_trip(self):
        tokenizer = BPETokenizer()
        training_text = "こんにちは世界。こんにちは。"
        tokenizer.train(training_text, vocab_size=30)

        encoded = tokenizer.encode("こんにちは世界")

        self.assertEqual(tokenizer.decode(encoded), "こんにちは世界")

    def test_save_and_load_preserve_behavior(self):
        tokenizer = BPETokenizer(special_tokens=["<bos>", "<eos>"])
        tokenizer.train("the quick brown fox jumps over the lazy dog", vocab_size=40)
        sample_text = "quick brown"
        encoded_before = tokenizer.encode(sample_text, add_bos=True, add_eos=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tokenizer.json"
            tokenizer.save(str(path))

            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            self.assertEqual(payload["version"], 2)
            self.assertEqual(payload["type"], "character_level_bpe")

            loaded = BPETokenizer.load(str(path))

        self.assertEqual(loaded.vocab_size, tokenizer.vocab_size)
        self.assertEqual(loaded.bos_token_id, tokenizer.bos_token_id)
        self.assertEqual(loaded.eos_token_id, tokenizer.eos_token_id)
        self.assertEqual(loaded.encode(sample_text, add_bos=True, add_eos=True), encoded_before)
        self.assertEqual(loaded.decode(encoded_before), sample_text)

    def test_empty_text_training_is_allowed(self):
        tokenizer = BPETokenizer()
        tokenizer.train("", vocab_size=0)

        self.assertEqual(tokenizer.merges, [])
        self.assertEqual(tokenizer.vocab_size, 0)
        self.assertEqual(tokenizer.encode(""), [])

    def test_too_small_vocab_size_raises(self):
        tokenizer = BPETokenizer()

        with self.assertRaises(ValueError):
            tokenizer.train("abc", vocab_size=2)

    def test_decode_unknown_token_id_raises(self):
        tokenizer = BPETokenizer()
        tokenizer.train("abcabc", vocab_size=10)

        with self.assertRaises(ValueError):
            tokenizer.decode([9999])

    def test_encode_unknown_character_raises(self):
        tokenizer = BPETokenizer()
        tokenizer.train("abcabc", vocab_size=10)

        with self.assertRaises(ValueError):
            tokenizer.encode("z")

    def test_special_token_insertion_and_skipping(self):
        tokenizer = BPETokenizer(special_tokens=["<bos>", "<eos>"])
        tokenizer.train("abcabcabc", vocab_size=10)

        encoded = tokenizer.encode("abc", add_bos=True, add_eos=True)

        self.assertEqual(encoded[0], tokenizer.bos_token_id)
        self.assertEqual(encoded[-1], tokenizer.eos_token_id)
        self.assertEqual(tokenizer.decode(encoded, skip_special_tokens=True), "abc")

    def test_decode_special_tokens_without_skipping_raises(self):
        tokenizer = BPETokenizer(special_tokens=["<bos>", "<eos>"])
        tokenizer.train("abcabcabc", vocab_size=10)
        encoded = tokenizer.encode("abc", add_bos=True)

        with self.assertRaises(ValueError):
            tokenizer.decode(encoded, skip_special_tokens=False)

    def test_load_rejects_malformed_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad_tokenizer.json"
            path.write_text('{"version": 2}', encoding="utf-8")

            with self.assertRaises(ValueError):
                BPETokenizer.load(str(path))

    def test_describe_merge_returns_readable_text(self):
        tokenizer = BPETokenizer()
        tokenizer.train("the the", vocab_size=10)

        description = tokenizer.describe_merge(0)

        self.assertIn("->", description)
        self.assertIn("+", description)
        self.assertTrue(description.startswith("'"))


if __name__ == "__main__":
    unittest.main()
