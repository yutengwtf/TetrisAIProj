from TetrisBrain import phi
from Hype import *
import torch 
import numpy as np
from collections import namedtuple, deque
import random

Transition = namedtuple( 'Transition', ('s1', 'a1', 'r1', 's2', 'done') )
Batch = namedtuple( 'Batch', ('s1_np', 'a1_np', 'r1_np', 's2_np', 'done_np'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_batch(buffer, bonus):
    transitions = buffer.sample(MINIBATCH_SIZE-1)
    transitions.append(bonus)
    s1_np = np.asarray([(t.s1) for t in transitions])
    a1_np = np.asarray([(t.a1) for t in transitions])
    r1_np = np.asarray([(t.r1) for t in transitions]).astype('float32')
    s2_np = np.asarray([(t.s2) for t in transitions])
    done_np = np.asarray([(t.done) for t in transitions])
    return Batch(s1_np, a1_np, r1_np, s2_np, done_np)

def train(brain, loss_function, brain_optim, batch, tb, epoch):
    y = (torch.as_tensor(batch.r1_np)).cuda() + (torch.as_tensor(1 - batch.done_np)).cuda() * DISCOUNT_FACTOR * brain.get_Vstar(phi(batch.s2_np))
    q = brain.get_Q(phi(batch.s1_np), phi(batch.a1_np).long())
    loss = loss_function(y, q)
    brain_optim.zero_grad()
    loss.backward()
    brain_optim.step()
    tb.add_scalar('Q', torch.mean(q).item(), epoch)
    tb.add_scalar('Loss', loss.item(), epoch)
