import torch
import torch.nn.functional as F
from enum import Enum
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from collections import namedtuple

from .samd_config import SamdConfig
from .draft import DraftModel, Candidates, CandidateType

from profile_utils import profile_decorator
    

class SamplingMethods(str, Enum):
    typical = "typical"
    nucleus = "nucleus"


class OptionalTensor:
    
    def __init__(self, data: Optional[torch.Tensor] = None):
        self.data = data

    def apply(self, fn: Callable) -> 'OptionalTensor':
        if self.data is None:
            return OptionalTensor(None)
        else:
            return OptionalTensor(fn(self.data))

@dataclass
class SamdGenerationConfig:
    max_new_tokens: int = field(default=512)
    max_cache_len: int = field(default=2048)


@profile_decorator("gen_candidates")
def gen_candidates(
    logits: torch.Tensor,
    tree_retrieve_indices: torch.Tensor,
    draft: DraftModel,
    samd_config: SamdConfig,
    gen_config: SamdGenerationConfig,
    device: torch.device,
):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - ...

    Returns:
    - tuple (torch.Tensor, List[int]): ...
    """
    # Greedy decoding: Select the most probable candidate from the original logits.
    start_token = torch.argmax(logits[:, -1]).item()
    candidate_type, tokens, buffers_kwargs = draft.lookup(start_token)
    tree_retrieve_indices = buffers_kwargs.get("tree_retrieve_indices", tree_retrieve_indices)
    if candidate_type == CandidateType.sequence_dyn \
        or candidate_type == CandidateType.sequence_static:
        tokens = torch.tensor([tokens], dtype=torch.long, device=device)
        candidate_tokens = tokens
    else:
        tokens_ext = torch.tensor(tokens + [0], dtype=torch.long, device=device)
        candidate_tokens = tokens_ext[tree_retrieve_indices]
        tokens = torch.tensor([tokens], dtype=torch.long, device=device)

    return Candidates(
        candidate_type,
        tokens,
        candidate_tokens,
        buffers_kwargs,
    )


@profile_decorator("eval_posterior")
def eval_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    config: SamdGenerationConfig,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    # Find the tokens that match the maximum logits for each position in the sequence
    posterior_mask = (
        candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
    ).int()
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
    accept_length = candidates_accept_length.max()
    # Choose the best candidate
    if accept_length == 0:
        # Default to the first candidate if none are accepted
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
    else:
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
    return best_candidate, accept_length + 1    
