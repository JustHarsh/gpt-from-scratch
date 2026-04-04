from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        chars = sorted(set(text))
        stoi = {}
        for i, c in enumerate(chars):
            stoi[c] = i

        itos = {c: i for i,c in stoi.items()}

        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        ret = []
        for char in text:
            ret.append(stoi[char])
        return ret

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        ret = []
        for id_ in ids:
            ret.append(itos[id_])
        return ''.join(ret)
