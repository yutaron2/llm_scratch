from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Tuple


class BPETokenizer:
    VERSION = 2
    TOKENIZER_TYPE = "character_level_bpe"

    def __init__(self, special_tokens: list[str] | None = None):
        if special_tokens is None:
            special_tokens = []

        if not isinstance(special_tokens, list):
            raise TypeError("special_tokens must be a list of strings")
        if not all(isinstance(token, str) for token in special_tokens):
            raise TypeError("special_tokens must contain only strings")
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special_tokens must be unique")

        self.special_tokens = list(special_tokens)
        self.base_vocab: List[str] = []
        self.special_token_to_id: Dict[str, int] = {}
        self.id_to_special_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self.merge_token_ids: Dict[Tuple[str, str], int] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._vocab_size = 0
        self._trained = False

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def bos_token_id(self) -> int:
        return self._get_special_token_id("<bos>")

    @property
    def eos_token_id(self) -> int:
        return self._get_special_token_id("<eos>")

    def train(self, text: str, vocab_size: int) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer")

        self.base_vocab = sorted(set(text))
        minimum_vocab_size = len(self.base_vocab) + len(self.special_tokens)
        if vocab_size < minimum_vocab_size:
            raise ValueError(
                f"vocab_size must be at least {minimum_vocab_size} "
                f"({len(self.base_vocab)} base characters + {len(self.special_tokens)} special tokens)"
            )

        self._reset_trained_state()
        self._initialize_token_maps()

        sequence = list(text)
        num_merges = vocab_size - minimum_vocab_size

        for _ in range(num_merges):
            pair_counts = Counter(zip(sequence, sequence[1:]))
            if not pair_counts:
                break

            best_pair, best_count = min(
                pair_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if best_count <= 1:
                break

            merged_token = "".join(best_pair)
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1

            if merged_token not in self.token_to_id:
                new_token_id = len(self.token_to_id)
                self.token_to_id[merged_token] = new_token_id
                self.id_to_token[new_token_id] = merged_token

            self.merge_token_ids[best_pair] = self.token_to_id[merged_token]
            sequence = self._replace_pair(sequence, best_pair, merged_token)

        self._vocab_size = len(self.token_to_id)
        self._trained = True

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        self._ensure_trained()

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        sequence = list(text)
        for character in sequence:
            if character not in self.token_to_id:
                raise ValueError(f"Unknown character: {character!r}")

        while True:
            best_index = None
            best_pair = None
            best_rank = None

            for index, pair in enumerate(zip(sequence, sequence[1:])):
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = index
                    best_pair = pair

            if best_index is None or best_pair is None:
                break

            merged_token_id = self.merge_token_ids[best_pair]
            merged_token = self.id_to_token[merged_token_id]
            sequence = sequence[:best_index] + [merged_token] + sequence[best_index + 2 :]

        token_ids = [self.token_to_id[token] for token in sequence]
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)

        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        self._ensure_trained()

        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list of integers")
        if not all(isinstance(token_id, int) for token_id in token_ids):
            raise TypeError("token_ids must contain only integers")

        pieces = []
        for token_id in token_ids:
            if token_id in self.id_to_special_token:
                if skip_special_tokens:
                    continue
                raise ValueError(
                    f"Cannot decode special token {self.id_to_special_token[token_id]!r} "
                    "when skip_special_tokens=False"
                )

            token = self.id_to_token.get(token_id)
            if token is None:
                raise ValueError(f"Unknown token id: {token_id}")
            pieces.append(token)

        return "".join(pieces)

    def save(self, path: str) -> None:
        self._ensure_trained()

        payload = {
            "version": self.VERSION,
            "type": self.TOKENIZER_TYPE,
            "special_tokens": self.special_tokens,
            "base_vocab": self.base_vocab,
            "merges": [list(pair) for pair in self.merges],
            "token_strings": {
                str(token_id): token
                for token_id, token in sorted(self.id_to_token.items())
            },
        }

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        cls._validate_payload(payload)

        tokenizer = cls(special_tokens=payload["special_tokens"])
        tokenizer.base_vocab = list(payload["base_vocab"])
        tokenizer._initialize_token_maps()
        tokenizer.merges = [tuple(pair) for pair in payload["merges"]]

        tokenizer.id_to_token = {
            int(token_id): token
            for token_id, token in payload["token_strings"].items()
        }
        tokenizer.token_to_id = {
            token: token_id for token_id, token in tokenizer.id_to_token.items()
        }
        tokenizer._rebuild_merge_metadata()
        tokenizer._vocab_size = len(tokenizer.token_to_id)
        tokenizer._trained = True
        return tokenizer

    def _reset_trained_state(self) -> None:
        self.merges = []
        self.merge_ranks = {}
        self.merge_token_ids = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_token_to_id = {}
        self.id_to_special_token = {}

    def _initialize_token_maps(self) -> None:
        for token_id, character in enumerate(self.base_vocab):
            self.token_to_id[character] = token_id
            self.id_to_token[token_id] = character

        special_start = len(self.base_vocab)
        self.special_token_to_id = {
            token: special_start + index
            for index, token in enumerate(self.special_tokens)
        }
        self.id_to_special_token = {
            token_id: token for token, token_id in self.special_token_to_id.items()
        }
        for token, token_id in self.special_token_to_id.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    @staticmethod
    def _replace_pair(
        sequence: list[str],
        pair: tuple[str, str],
        merged_token: str,
    ) -> list[str]:
        merged_sequence = []
        index = 0

        while index < len(sequence):
            if (
                index < len(sequence) - 1
                and sequence[index] == pair[0]
                and sequence[index + 1] == pair[1]
            ):
                merged_sequence.append(merged_token)
                index += 2
            else:
                merged_sequence.append(sequence[index])
                index += 1

        return merged_sequence

    def _rebuild_merge_metadata(self) -> None:
        self.merge_ranks = {}
        self.merge_token_ids = {}

        for rank, pair in enumerate(self.merges):
            merged_token = "".join(pair)
            token_id = self.token_to_id.get(merged_token)
            if token_id is None:
                token_id = len(self.token_to_id)
                self.token_to_id[merged_token] = token_id
                self.id_to_token[token_id] = merged_token
            self.merge_ranks[pair] = rank
            self.merge_token_ids[pair] = token_id

    def _get_special_token_id(self, token: str) -> int:
        self._ensure_trained()
        token_id = self.special_token_to_id.get(token)
        if token_id is None:
            raise ValueError(f"Special token {token!r} is not configured for this tokenizer")
        return token_id

    def _ensure_trained(self) -> None:
        if not self._trained:
            raise ValueError("Tokenizer must be trained or loaded before encode/decode/save")

    @classmethod
    def _validate_payload(cls, payload: object) -> None:
        if not isinstance(payload, dict) or not payload:
            raise ValueError("Tokenizer JSON must contain an object payload")

        required_keys = {"version", "type", "special_tokens", "base_vocab", "merges", "token_strings"}
        missing_keys = required_keys.difference(payload.keys())
        if missing_keys:
            missing_keys_text = ", ".join(sorted(missing_keys))
            raise ValueError(f"Tokenizer JSON is missing required keys: {missing_keys_text}")

        if payload["version"] != cls.VERSION:
            raise ValueError(f"Unsupported tokenizer version: {payload['version']}")
        if payload["type"] != cls.TOKENIZER_TYPE:
            raise ValueError(f"Unsupported tokenizer type: {payload['type']}")

        special_tokens = payload["special_tokens"]
        if not isinstance(special_tokens, list) or not all(
            isinstance(token, str) for token in special_tokens
        ):
            raise ValueError("special_tokens must be a list of strings")
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special_tokens must be unique")

        base_vocab = payload["base_vocab"]
        if not isinstance(base_vocab, list) or not all(
            isinstance(token, str) and len(token) == 1 for token in base_vocab
        ):
            raise ValueError("base_vocab must be a list of single-character strings")
        if len(set(base_vocab)) != len(base_vocab):
            raise ValueError("base_vocab must be unique")

        merges = payload["merges"]
        if not isinstance(merges, list):
            raise ValueError("merges must be a list")
        for pair in merges:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(token, str) for token in pair)
            ):
                raise ValueError("Each merge must be a list of two strings")

        token_strings = payload["token_strings"]
        if not isinstance(token_strings, dict):
            raise ValueError("token_strings must be an object")

        for token_id_text, token in token_strings.items():
            try:
                int(token_id_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid token id key: {token_id_text!r}") from exc
            if not isinstance(token, str):
                raise ValueError(f"token_strings[{token_id_text!r}] must be a string")

    def describe_merge(self, index: int) -> str:
        self._ensure_trained()
        if not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0 or index >= len(self.merges):
            raise IndexError("merge index out of range")

        left, right = self.merges[index]
        return f"{left!r} + {right!r} -> {(left + right)!r}"
