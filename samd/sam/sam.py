import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm import tqdm

class SAM:
    
    @dataclass
    class SamplingState:
        topk_tokens: List[Tuple[int, float]]
    
    @dataclass
    class SAMState:
        next: dict[int, int]
        link: int
        length: int
        endpos_cnt: int

    @staticmethod
    def build(
        batch_tokens: List[List[int]], 
        eos_token: int,
        n_gram: int,
        k: int,
        verbose: bool =True
    ):
        sam = SAM(n_gram=n_gram, k=k)
        sam.build_states(batch_tokens, eos_token, verbose)
        sam.count_endpos(batch_tokens, eos_token, verbose)
        sam.build_sampling_states(k, eos_token, verbose)
        sam.state_pruning(n_gram, verbose)
        return sam

    def __init__(self, n_gram: int, k: int):
        self.n_gram = n_gram
        self.k = k
        self.states: List[SAM.SAMState] = [SAM.SAMState(next={}, link=-1, length=0, endpos_cnt=0)]
        self.sampling_states: List[SAM.SamplingState] = None
        self.last = 0
        
        # params needed to be reset for each query
        self.state_index = 0
        self.hot_states: Dict[int, SAM.SamplingState] = {}

    def add_token(self, token: int):
        cur = len(self.states)
        self.states.append(self.SAMState(next={}, link=-1, length=self.states[self.last].length + 1, endpos_cnt=0))
        p = self.last
        while p != -1 and token not in self.states[p].next:
            self.states[p].next[token] = cur
            p = self.states[p].link
        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].next[token]
            if self.states[p].length + 1 == self.states[q].length:
                self.states[cur].link = q
            else:
                clone = len(self.states)
                self.states.append(deepcopy(self.states[q]))
                self.states[clone].length = self.states[p].length + 1
                while p != -1 and self.states[p].next[token] == q:
                    self.states[p].next[token] = clone
                    p = self.states[p].link
                self.states[q].link = self.states[cur].link = clone
        self.last = cur
    
    def build_states(self, batch_tokens: List[List[int]], eos_token: int, verbose: bool):
        for tokens in tqdm(batch_tokens, desc="build sam states...", disable=not verbose):
            for token in tokens:
                self.add_token(token)
            if tokens[-1] != eos_token:
                self.add_token(eos_token)

    def count_endpos(self, batch_tokens: List[List[int]], eos_token: int, verbose: bool):
        index = 0
        for tokens in tqdm(batch_tokens, desc="count endpos...", disable=not verbose):
            for token in tokens:
                index = self.states[index].next[token]
                self.states[index].endpos_cnt += 1
            if tokens[-1] != eos_token:
                index = self.states[index].next[eos_token]
                self.states[index].endpos_cnt += 1
        childs_cnt = [0 for _ in range(len(self.states))]
        for i in range(1, len(self.states)):
            childs_cnt[self.states[i].link] += 1
        q = deque([i for i in range(len(self.states)) if childs_cnt[i] == 0])
        while len(q) != 0:
            u = q.popleft()
            if u == 0:
                continue
            v = self.states[u].link
            self.states[v].endpos_cnt += self.states[u].endpos_cnt
            childs_cnt[v] -= 1
            if childs_cnt[v] == 0:
                q.append(v)
    
    def build_sampling_states(self, k: int, eos_token: int, verbose: bool):
        self.sampling_states = [SAM.SamplingState(topk_tokens=[(None, None)] * k) for _ in range(len(self.states))]
        childs = [[] for _ in range(len(self.states))]
        for state_id, state in enumerate(self.states):
            if state.link != -1:
                childs[state.link].append(state_id)
        
        index_list: List[int] = []
        dq = deque([0])
        while len(dq) != 0:
            u = dq.popleft()
            index_list.append(u)
            for v in childs[u]:
                dq.append(v)
        
        for i in tqdm(index_list, desc="build sampling states...", disable=not verbose, total=len(self.states)):
            state, samping_state = self.states[i], self.sampling_states[i]
            topk = sorted(
                [(token, self.states[next_id].endpos_cnt / state.endpos_cnt) for token, next_id in state.next.items()],
                key=lambda item: item[1],
                reverse=True
            )[:k]
            if len(topk) == 0:
                topk.append((eos_token, 0.0))
            samping_state.topk_tokens = topk

    def state_pruning(self, n_gram: int, verbose: bool):
        mask = [True] * len(self.states)
        for i, state in tqdm(enumerate(self.states), 
                             desc="state pruning...", disable=not verbose, total=len(self.states)):
            if state.link == -1:
                continue
            if self.states[state.link].length + 1 > n_gram:
                mask[i] = False
        id_map = {}
        for i in range(len(mask)):
            if mask[i]:
                id_map[i] = len(id_map)
        id_map[-1] = -1
        new_states = [self.states[i] for i in range(len(self.states)) if mask[i]]
        new_sampling_states = [self.sampling_states[i] for i in range(len(self.states)) if mask[i]]
        for state in new_states:
            state.link = id_map[state.link]
            state.next = {k: id_map[v] for k, v in state.next.items() if v in id_map}
        self.states = new_states
        self.sampling_states = new_sampling_states

    def reset_state(self):
        self.state_index = 0
    
    def transfer_state(self, tokens: List[int]):
        for token in tokens:
            self.state_index = self.get_next_index(self.state_index, token)
    
    def get_next_index(self, cur_index: int, token: int):
        state = self.states[cur_index]
        while token not in state.next and cur_index != 0:
            cur_index = state.link
            state = self.states[cur_index]
        next_index = state.next.get(token, 0)
        return next_index

    def get_tree_tokens(self, start_token: int, tree: List[List[int]]):
        tree_tokens: List[int] = [start_token] + [None] * (len(tree) - 1)
        tree_indices: List[int] = [None] * len(tree)
        cur_index = self.get_next_index(self.state_index, start_token)
        tree_indices[0] = cur_index
        for node_id, childs in enumerate(tree):
            cur_token = tree_tokens[node_id]
            cur_index = tree_indices[node_id]
            topk_tokens = self.get_merged_topk_tokens(cur_token, cur_index)
            for child_id, child in enumerate(childs):
                token = topk_tokens[child_id][0]
                next_index = self.get_next_index(cur_index, token)
                tree_tokens[child] = token
                tree_indices[child] = next_index
        return tree_tokens, tree_indices
    
    def get_merged_topk_tokens(self, cur_token: int, cur_index: int):    
        if cur_token not in self.hot_states:
            topk_tokens = self.sampling_states[cur_index].topk_tokens
        else:
            tokens = self.hot_states[cur_token].topk_tokens + self.sampling_states[cur_index].topk_tokens
            tokens = sorted(tokens, key=lambda item: item[1], reverse=True)
            topk_tokens = tokens[:self.k]
        if len(topk_tokens) < self.k:
            n = self.k - len(topk_tokens)
            topk_tokens = topk_tokens + [topk_tokens[-1]] * n
        return topk_tokens
    
    def set_state_index(self, index: int):
        self.state_index = index
    
    def set_k(self, k: int):
        self.k = k
        for state in self.sampling_states:
            state.topk_tokens = state.topk_tokens[:k]

    def update_samping_state(self, state_indices: List[int], path_topk: List[List[Tuple[int, float]]]):
        for state_index, topk_tokens in zip(state_indices, path_topk):
            self.sampling_states[state_index].topk_tokens = topk_tokens

    def update_hot_state(self, tokens: List[int], path_topk: List[List[Tuple[int, float]]]):
        for token, topk_tokens in zip(tokens, path_topk):
            self.hot_states[token] = SAM.SamplingState(topk_tokens=topk_tokens)

    def print_state(self):
        print("state_index:", self.state_index)
        print("length: {}, endpos_cnt: {}".format(
            self.states[self.state_index].length, 
            self.states[self.state_index].endpos_cnt,
        ))
