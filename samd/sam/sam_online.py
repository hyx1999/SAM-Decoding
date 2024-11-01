import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm import tqdm

class SAMOnline:
    
    @dataclass
    class SamplingState:
        topk_tokens: List[int]
    
    @dataclass
    class SAMState:
        next: dict[int, int]
        link: int
        length: int
        endpos_cnt: int

    def __init__(self, n_gram: int, k: int):
        self.n_gram = n_gram
        self.k = k
        self.states: List[SAMOnline.SAMState] = [self.SAMState(next={}, link=-1, length=0, endpos_cnt=0)]
        self.sampling_states: List[SAMOnline.SamplingState] = [self.SamplingState(topk_tokens=[])]
        self.last = 0
        
        # params needed to be reset for each query
        self.cur_index = 0
        self.cur_length = 0
    
    def expand_state(self, state: SAMState):
        new_state_id = len(self.states)
        self.states.append(state)
        self.sampling_states.append(self.SamplingState(topk_tokens=[]))
        return new_state_id

    def add_state(self, token: int):
        cur = self.expand_state(self.SAMState(next={}, link=-1, length=self.states[self.last].length + 1, endpos_cnt=0))
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
                clone = self.expand_state(deepcopy(self.states[q]))
                self.states[clone].length = self.states[p].length + 1
                self.sampling_states[clone] = self.sampling_states[p]
                while p != -1 and self.states[p].next[token] == q:
                    self.states[p].next[token] = clone
                    p = self.states[p].link
                self.states[q].link = self.states[cur].link = clone
        self.last = cur
    
    def update_state(self, index: int, token: int):
        topk_tokens = self.sampling_states[index].topk_tokens
        if token in topk_tokens:
            topk_tokens.remove(token)
        topk_tokens.insert(0, token)
        if len(topk_tokens) > self.k:
            topk_tokens.pop(-1)
    
    def update_state_recursive(self, index: int, token: int):
        while True:
            self.update_state(index, token)
            index = self.states[index].link
            if index == 0:
                break
    
    def get_next_index(self, index: int, length: int, token: int, cutoff: bool = True):
        while index != 0 and token not in self.states[index].next:
            index = self.states[index].link
            length = self.states[index].length
        if token in self.states[index].next:
            index = self.states[index].next[token]
            length += 1
        else:
            index = length = 0
        if cutoff:
            while length > self.n_gram:
                index = self.states[index].link
                length = self.states[index].length
        return index, length
    
    def add_tokens(self, tokens: List[int]):
        for token in tokens:
            self.add_state(token)
        for token in tokens:
            self.update_state_recursive(self.cur_index, token)
            self.cur_index, self.cur_length \
                = self.get_next_index(self.cur_index, self.cur_length, token)
        if self.cur_index == self.last:
            self.cur_index = self.states[self.cur_index].link
            self.cur_length = self.states[self.cur_index].length
