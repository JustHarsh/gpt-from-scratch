# import re
# import torch
# import torch.nn as nn
# from torchtyping import TensorType
# from typing import List


# class Solution:

#     def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:

#         def preprocess(sentence: str) -> list[str]:
#             clean = re.sub(r'[^\w\s]', '', sentence)
#             return [w.lower() for w in clean.split()]

#         all_sentences = positive + negative

#         # Build vocab
#         vocabulary = set()
#         for sentence in all_sentences:
#             vocabulary.update(preprocess(sentence))

#         word_to_id = {word: i + 1 for i, word in enumerate(sorted(vocabulary))}

#         # Encode
#         tensors = [
#             torch.tensor([word_to_id[w] for w in preprocess(s)], dtype=torch.float)
#             for s in all_sentences
#         ]

#         # Pad
#         return nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0)


import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        combined = positive + negative

        # Build vocabulary: sorted unique words -> integer IDs starting at 1
        vocabulary = sorted({word for sentence in combined for word in sentence.split()})
        word_to_id = {word: idx + 1 for idx, word in enumerate(vocabulary)}

        # Encode each sentence as a tensor of word IDs
        encoded = [torch.tensor([word_to_id[w] for w in s.split()]) for s in combined]

        # Pad shorter sequences with 0s so output is a rectangular tensor
        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)