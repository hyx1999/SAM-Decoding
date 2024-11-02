import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm import tqdm

class SAMOnline:
   
    @dataclass
    class SAMState:
        next: dict[int, int]
        link: int
        length: int
        endpos: int

    def __init__(self, n_predict: int = 9):
        self.n_predict = n_predict
        self.states: List[SAMOnline.SAMState] = [self.SAMState(next={}, link=-1, length=0, endpos=0)]
        self.input_ids: List[int] = [-1]
        self.last = 0
        self.max_length = 0
        
        # params needed to be reset for each query
        self.cur_index = 0
        self.cur_length = 0
    
    def expand_state(self, state: SAMState):
        new_index = len(self.states)
        self.states.append(state)
        return new_index

    def add_state(self, token: int):
        self.max_length += 1
        cur = self.expand_state(
            self.SAMState(
                next={}, link=-1, 
                length=self.max_length, 
                endpos=self.max_length
            )
        )
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
                self.states[clone].endpos = self.states[q].endpos
                while p != -1 and self.states[p].next[token] == q:
                    self.states[p].next[token] = clone
                    p = self.states[p].link
                self.states[q].link = self.states[cur].link = clone
        self.last = cur
           
    def transfer_state(self, index: int, length: int, token: List[int]):
        while index != 0 and token not in self.states[self.cur_index].next:
            index = self.states[index].link
            length = self.states[index].length
        if token in self.states[index].next:
            index = self.states[index].next[token]
            length += 1
        else:
            index = length = 0
        return index, length
    
    def transfer_cur_state(self, token: int):
        self.cur_index, self.cur_length = \
            self.transfer_state(self.cur_index, self.cur_length, token)
    
    def to_anc(self, index: int, length: int):
        while index != 0 and length + self.n_predict > self.max_length:
            index = self.states[index].link
            length = self.states[index].length
        return index, length
    
    def add_tokens(self, tokens: List[int]):
        for token in tokens:
            self.add_state(token)
            self.transfer_cur_state(token)
        self.input_ids.extend(tokens)
        self.cur_index, self.cur_length = \
            self.to_anc(self.cur_index, self.cur_length)

    def lookup(self, token: int):
        index, length = \
            self.transfer_state(self.cur_index, self.cur_length, token)
        index, length = \
            self.to_anc(index, length)
        endpos = self.states[index].endpos
        pred_ids = self.input_ids[endpos + 1:endpos + self.n_predict + 1]
        if length == 0:
            pred_ids = [-1] * self.n_predict
        if len(pred_ids) < self.n_predict:
            pred_ids.extend([-1] * (self.n_predict - len(pred_ids)))
        return pred_ids
