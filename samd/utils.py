import torch
import torch.nn.functional as F
from enum import Enum
from typing import List, Dict
from dataclasses import dataclass, field
from collections import namedtuple

from .samd_config import SamdConfig
from .sam import SAM
    

class SamplingMethods(str, Enum):
    typical = "typical"
    nucleus = "nucleus"

Candidates = namedtuple('tree_candidates', ['tree_tokens', 'candidate_tokens', 'candidate_indices', 'candidate_sam_indices'])

@dataclass
class SamdGenerationConfig:
    max_new_tokens: int = field(default=512)
    max_cache_len: int = field(default=1024)
    temperature: float = field(default=0.)
    posterior_threshold: float = field(default=0.3)
    posterior_alpha: float = field(default=0.09)
    top_p: float = field(default=0.8) 
    sampling: str = field(default='typical', metadata={"choices": [SamplingMethods.typical, SamplingMethods.nucleus]})
    fast: bool = field(default=False)


def pad_path(path, length, pad_value=-1):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-1): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -1, -1]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def gen_sd_buffers(
    tree: List[List[int]], 
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Generate buffers for the SD based on the provided bfs tree.
    
    Parameters:
    - tree (List[List[int]]): A nested list represent the SD tree structure.
    - device (torch.device): The device to save the tensors.
    
    Returns:
    - dict: A dictionary containing buffers related to the SD structure.
    """
    num_nodes = len(tree)
    
    anc_dict = {0: -1}
    for node_id, childs in enumerate(tree):
        for child in childs:
            anc_dict[child] = node_id

    level_dict = {0: 0}
    for node_id in range(1, num_nodes):
        level_dict[node_id] = level_dict[anc_dict[node_id]] + 1
    
    # Create the attention mask for Medusa
    samd_attn_mask = torch.eye(num_nodes, num_nodes)
    for node_id in range(num_nodes):
        ancs = [node_id]
        x = node_id
        while x != -1:
            ancs.append(x)
            x = anc_dict[x]
        ancs = torch.tensor(ancs, dtype=torch.long)
        samd_attn_mask[node_id, ancs] = True
    samd_attn_mask = samd_attn_mask.view(1, 1, num_nodes, num_nodes)
    
    samd_position_ids = torch.zeros((1, num_nodes), dtype=torch.long)
    for i in range(num_nodes):
        samd_position_ids[:, i] = level_dict[i]

    sd_buffers = {
        "samd_attn_mask": samd_attn_mask,
        "samd_position_ids": samd_position_ids,
    }
    
    sd_buffers = {k: v.to(device) for k, v in sd_buffers.items()}
    return sd_buffers


def get_nucleus_one_token(logits, config: SamdGenerationConfig):
    """
    Performs token sampling based on the nucleus (top-p) sampling method.

    This function selects a token from a given logit distribution using the nucleus sampling strategy.
    It allows for more controlled and diverse generation compared to traditional top-k sampling.

    Args:
        logits (torch.Tensor): The logits from a language model output, expected to be a 2D tensor (BxC).
        temperature (float): A temperature parameter to control the randomness in sampling.
                             Higher values increase diversity, lower values make selections more deterministic.
        top_p (float): The cumulative probability threshold for nucleus sampling.
                       It controls the size of the set of high-probability tokens to consider for sampling.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    if config.top_p >= 1:
        return torch.multinomial(F.softmax(logits / config.temperature, dim=-1), 1)
    logits = logits / config.temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_logits, dim=-1)
    sorted_indices_to_remove = cum_probs > config.top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    sampled_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
    return sampled_token


def get_typical_one_token(logits, config: SamdGenerationConfig):
    """
    Implements token sampling based on the typical sampling method.

    This function selects a token from a given logit distribution using the typical sampling strategy,
    aiming to balance between diversity and likelihood in a more nuanced way compared to traditional methods.

    Args:
        logits (torch.Tensor): The logits from a language model output, expected to be a 2D tensor.
        temperature (float): A parameter to control the randomness in sampling.
                              Higher values increase diversity, lower values make selections more deterministic.
        posterior_threshold (float): A threshold to decide the lower bound of probabilities to be considered for sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    logits = logits / config.temperature
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * config.posterior_threshold,
            torch.exp(-entropy) * config.posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logits[indices_to_remove] = float('-inf')
    sampled_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
    return sampled_token


def gen_candidates(
    logits,
    sam: SAM,
    sd_config: SamdConfig,
    gen_config: SamdGenerationConfig,
    eos_token: int,
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
    if gen_config.temperature == 0 or gen_config.fast:
        start_token = torch.argmax(logits[:, -1]).item()
    else:
        if gen_config.sampling == SamplingMethods.typical:
            start_token = get_typical_one_token(logits[:, -1], gen_config).item()
        elif gen_config.sampling == SamplingMethods.nucleus:
            start_token = get_nucleus_one_token(logits[:, -1], gen_config).item()
        else:
            raise NotImplementedError
    tree_tokens, tree_indices = sam.get_tree_tokens(start_token, sd_config.tree)
    candidate_tokens = [[tree_tokens[0]]] + [None] * (len(sd_config.tree) - 1)
    candidate_indices = [[0]] + [None] * (len(sd_config.tree) - 1)
    candidate_sam_indices = [[tree_indices[0]]] + [None] * (len(sd_config.tree) - 1)
    # candidate_indices => sam_indices
    for node_id, childs in enumerate(sd_config.tree):
        for child_id in childs:
            candidate_tokens[child_id] = candidate_tokens[node_id] + [tree_tokens[child_id]]
            candidate_indices[child_id] = candidate_indices[node_id] + [child_id]
            candidate_sam_indices[child_id] = candidate_sam_indices[node_id] + [tree_indices[child_id]]
    
    max_level = max(len(p) for p in candidate_tokens)
    candidate_tokens = [pad_path(p, max_level, -1) for p in candidate_tokens]
    candidate_indices = [pad_path(p, max_level, -1) for p in candidate_indices]
    candidate_sam_indices = [pad_path(p, max_level, -1) for p in candidate_sam_indices]
    
    tree_tokens = torch.tensor(tree_tokens, dtype=torch.long, device=device).unsqueeze(0)
    candidate_tokens = torch.tensor(candidate_tokens, dtype=torch.long, device=device)
    candidate_indices = torch.tensor(candidate_indices, dtype=torch.long, device=device)
    return Candidates(
        tree_tokens, 
        candidate_tokens, 
        candidate_indices, 
        candidate_sam_indices
    )


def get_nucleus_posterior_mask(logits: torch.Tensor, candidates: torch.Tensor, config: SamdGenerationConfig):
    """
    Generates a posterior mask for token candidates using nucleus (top-p) sampling.

    This function applies nucleus sampling to a set of logits, and then generates a mask indicating 
    which candidate tokens are selected. It adapts the sampling strategy to accommodate for 
    temperature scaling and cumulative probability thresholding.

    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / config.temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    if config.top_p >= 1:
        sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
        sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
        posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
        return posterior_mask
    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > config.top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

    
    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')
    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask


def get_typical_posterior_mask(logits: torch.Tensor, candidates: torch.Tensor, config: SamdGenerationConfig):
    """
    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        posterior_threshold (float): The minimum threshold for probabilities to be considered in sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    logits = logits[:, :-1] / config.temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * config.posterior_threshold,
            torch.exp(-entropy) * config.posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logits[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
    return posterior_mask


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
    if config.temperature == 0:
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
    elif config.sampling == SamplingMethods.typical:
        if config.fast:
            posterior_prob = torch.softmax(logits[:, :-1] / config.temperature, dim=-1)
            candidates_prob = torch.gather(
                posterior_prob, dim=-1, index=candidates[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            posterior_entropy = -torch.sum(
                posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
            )  # torch.sum(torch.log(*)) is faster than torch.prod
            threshold = torch.minimum(
                torch.ones_like(posterior_entropy) * config.posterior_threshold,
                torch.exp(-posterior_entropy) * config.posterior_alpha,
            )
            posterior_mask = candidates_prob > threshold
            candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)

            # Choose the best candidate based on the evaluated posterior probabilities
            accept_length = candidates_accept_length.max()
            if accept_length == 0:
                # If no candidates are accepted, just choose the first one
                best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            else:
                best_candidates = torch.where(candidates_accept_length == accept_length)[0]
                # Accept the best one according to likelihood
                likelihood = torch.sum(
                    torch.log(candidates_prob[best_candidates, :accept_length]), dim=-1
                )
                best_candidate = best_candidates[torch.argmax(likelihood)]
            return best_candidate, accept_length + 1
        else:
            # Calculate posterior probabilities and thresholds for candidate selection
            posterior_mask = get_typical_posterior_mask(logits, candidates, config)
            candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
            # Choose the best candidate based on the evaluated posterior probabilities
            accept_length = candidates_accept_length.max()
            
            if accept_length == 0:
                # If no candidates are accepted, just choose the first one
                best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            else:
                best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
                # Accept the best one according to likelihood
            return best_candidate, accept_length + 1
    elif config.sampling == SamplingMethods.nucleus:
        assert config.top_p < 1.0 + 1e-6, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, config)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length + 1
    else:
        raise NotImplementedError
