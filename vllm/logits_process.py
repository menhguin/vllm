from typing import Callable, List, Protocol, Tuple, Union, runtime_checkable
from abc import ABC, abstractmethod

import torch

from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer

@runtime_checkable
class LogitsProcessor(Protocol):
    """Protocol for all logit processors that can be applied during generation."""
    def __call__(self, past_tokens: List[int], logits: torch.Tensor) -> torch.Tensor:
        """Process input logits to generate modified logits for sampling."""
        ...

class LogitsProcessorBase(ABC):
    """Abstract base class for all logit processors that can be applied during generation."""
    
    @abstractmethod
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Process input logits to generate modified logits for sampling."""
        pass

def get_bad_words_logits_processors(
        bad_words: List[str],
        tokenizer: AnyTokenizer) -> List[LogitsProcessor]:
    bad_words_ids: List[List[int]] = list()

    for bad_word in bad_words:
        # To prohibit words both at the beginning
        # and in the middle of text
        # (related to add_prefix_space tokenizer parameter)
        for add_prefix_space in [False, True]:
            prefix = " " if add_prefix_space else ""
            prompt = prefix + bad_word.lstrip()

            if isinstance(tokenizer, MistralTokenizer):
                # Mistral tokenizers should not add special tokens
                prompt_token_ids = tokenizer.encode(prompt=prompt)
            else:
                prompt_token_ids = tokenizer.encode(text=prompt,
                                                    add_special_tokens=False)

            # If no space at the beginning
            # or if prefix space produces a new word token
            if (not add_prefix_space) or (
                    add_prefix_space
                    and prompt_token_ids[0] != bad_words_ids[-1][0]
                    and len(prompt_token_ids) == len(bad_words_ids[-1])):
                bad_words_ids.append(prompt_token_ids)

    return [NoBadWordsLogitsProcessor(bad_words_ids=bad_words_ids)]


class NoBadWordsLogitsProcessor:
    _SMALLEST_LOGIT = float("-inf")
    _NEUTRAL_LOGIT = 0.0

    def __init__(self, bad_words_ids: List[List[int]]):
        self.bad_words_ids = bad_words_ids
        self.word_bias: torch.FloatTensor = None

    def __call__(
        self,
        past_tokens_ids: Union[List[int], Tuple[int]],
        logits: torch.FloatTensor,
    ) -> torch.Tensor:
        if self.word_bias is None:
            self._init_word_bias(logits=logits)

        last_token_bias = torch.zeros_like(logits)

        for bad_word_ids in self.bad_words_ids:
            if len(bad_word_ids) == 1:  # 1-token words already processed
                continue

            if len(bad_word_ids) > len(past_tokens_ids) + 1:
                continue

            prefix_length = len(bad_word_ids) - 1
            last_token_id = bad_word_ids[-1]
            actual_prefix = past_tokens_ids[-prefix_length:]
            expected_prefix = bad_word_ids[:prefix_length]

            assert len(actual_prefix) == len(expected_prefix)

            is_match = tuple(actual_prefix) == tuple(expected_prefix)
            last_token_bias[last_token_id] += (self._SMALLEST_LOGIT if is_match
                                               else self._NEUTRAL_LOGIT)

        logits = logits + self.word_bias + last_token_bias

        return logits

    def _init_word_bias(self, logits: torch.FloatTensor) -> None:
        # Code based on NoBadWordsLogitsProcessor and SequenceBiasLogitsProcessor  # noqa: E501
        # from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py

        vocab_size = logits.shape[-1]

        self._check_token_ids_bounds(vocab_size=vocab_size)

        self.word_bias = torch.zeros((vocab_size, ),
                                     dtype=torch.float,
                                     device=logits.device)

        for bad_word_ids in self.bad_words_ids:
            if len(bad_word_ids) == 1:
                bad_word_id = bad_word_ids[-1]
                self.word_bias[bad_word_id] = self._SMALLEST_LOGIT

    def _check_token_ids_bounds(self, vocab_size: int) -> None:
        invalid_token_ids = []

        for bad_word_ids in self.bad_words_ids:
            for token_id in bad_word_ids:
                if token_id < 0 or token_id >= vocab_size:
                    invalid_token_ids.append(token_id)

        if len(invalid_token_ids) > 0:
            raise ValueError(
                f"The model vocabulary size is {vocab_size},"
                f" but the following tokens"
                f" were specified as bad: {invalid_token_ids}."
                f" All token id values should be integers satisfying:"
                f" 0 <= token_id < {vocab_size}.")


class MinZLogitsProcessor:
    """Processor that implements min-z sampling."""
    
    def __init__(self, min_z: float, min_tokens: int = 1):
        self.min_z = min_z
        self.min_tokens = min_tokens

    def __call__(self, past_tokens: List[int], logits: torch.Tensor) -> torch.Tensor:
        """Apply min-z sampling to the logits.
        
        Args:
            past_tokens: List of previously generated tokens (unused in min-z)
            logits: Logits to be processed
            
        Returns:
            Modified logits after applying min-z sampling
        """
        return _apply_min_z(logits, self.min_z, self.min_tokens)

def _apply_min_z(
    logits: torch.Tensor,
    min_z: float,
    min_tokens: int = 1
) -> torch.Tensor:
    if min_z <= 0.0:
        return logits

    # Get probs
    probs = torch.softmax(logits, dim=-1)
    
    # Calculate statistics
    max_probs, _ = probs.max(dim=-1, keepdim=True)
    median_probs = torch.median(probs, dim=-1, keepdim=True).values
    std_probs = torch.clamp(probs.std(dim=-1, keepdim=True), min=1e-9)
    
    # Compute z-scores
    z_scores = (probs - median_probs) / std_probs
    max_z = (max_probs - median_probs) / std_probs
    
    # Apply threshold
    scaled_min_z = min_z * max_z
    tokens_to_remove = z_scores < scaled_min_z

    # Ensure at least min_tokens are kept
    if min_tokens > 1:
        row_sums = (~tokens_to_remove).sum(dim=-1)
        need_adjust = row_sums < min_tokens
        if need_adjust.any():
            topk_values = torch.topk(logits[need_adjust], k=min_tokens, dim=-1)[1]
            tokens_to_remove[need_adjust].scatter_(1, topk_values, False)
    
    logits = logits.masked_fill(tokens_to_remove, float("-inf"))
    return logits
