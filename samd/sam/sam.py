from typing import List, Tuple
from dataclasses import dataclass
from copy import deepcopy
from collections import deque

class SAM:
    
    @dataclass
    class SamplingState:
        topk_tokens: List[int]
    
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
    ):
        sam = SAM()
        sam.build_states(batch_tokens, eos_token)
        sam.count_endpos(batch_tokens, eos_token)
        sam.build_sampling_states(k)
        sam.state_pruning(n_gram)
        return sam

    def __init__(self, n_gram: int, k: int):
        self.n_gram = n_gram
        self.k = k
        self.states: List[SAM.SAMState] = [SAM.SAMState(next={}, link=-1, length=0, endpos_cnt=0)]
        self.sampling_states: List[SAM.SamplingState] = None
        self.last = 0
        self.state_index = 0

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
    
    def build_states(self, batch_tokens: List[List[int]], eos_token: int):
        for tokens in batch_tokens:
            for token in tokens:
                self.add_token(token)
            if tokens[-1] != eos_token:
                self.add_token(eos_token)

    def count_endpos(self, batch_tokens: List[List[int]], eos_token: int):
        index = 0
        for tokens in batch_tokens:
            for token in tokens + [eos_token]:
                index = self.states[index].next[token]
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
    
    def build_sampling_states(self, k: int):
        self.sampling_states = [SAM.SamplingState(topk_tokens=[None] * k) for _ in range(len(self.states))]
        for state, samping_state in zip(self.states, self.sampling_states):
            sorted_tokens = sorted([(self.states[s_id].endpos_cnt, token) for token, s_id in state.next.items()], reverse=True)
            topk_tokens = [token for _, token in sorted_tokens[:k]]
            if len(topk_tokens) < k:
                f_sampling_state = self.sampling_states[state.link]
                for token in f_sampling_state.topk_tokens:
                    if token not in topk_tokens:
                        topk_tokens.append(token)
                    if len(topk_tokens) == k:
                        break
            samping_state.topk_tokens = topk_tokens

    def state_pruning(self, n_gram: int):
        mask = [True] * len(self.states)
        for i, state in enumerate(self.states):
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
        while token not in state.next:
            cur_index = state.link
            state = self.states[cur_index]
        next_index = state.next[token]
        return next_index
        
    def get_tree_tokens(self, start_token: int, tree: List[List[int]]):
        tree_tokens: List[int] = [start_token] + [None] * (len(tree) - 1)
        tree_indices: List[int] = [None] * len(tree)
        cur_index = self.get_next_index(self.state_index, start_token)
        tree_indices[0] = cur_index
        for node_id, childs in enumerate(tree):
            cur_index = tree_indices[node_id]
            for child_id, child in enumerate(childs):
                token = self.sampling_states[cur_index].topk_tokens[child_id]
                next_index = self.get_next_index(cur_index, token)
                tree_tokens[child] = token
                tree_indices[child] = next_index
        return tree_tokens, tree_indices

    def update_samping_state(self, state_indices: List[int], path_topk: List[List[int]]):
        for state_index, topk_tokens in zip(state_indices, path_topk):
            self.sampling_states[state_index].topk_tokens = topk_tokens
