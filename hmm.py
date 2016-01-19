#!/usr/bin/env python2
# -*- coding:utf-8 -*-
import numpy as np

class HMM(object):
    def __init__(self, initial_dist, transition_probs, emission_probs):
        self.transition_probs = transition_probs #A
        self.emission_probs = emission_probs     #B
        self.initial_dist = initial_dist         #pi

    def get_emmision_dist(self, emission):
        return self.emission_probs[:, emission]

    @property
    def states_count(self):
        return self.transition_probs.shape[0]

    @property
    def emission_count(self):
        return self.emission_probs.shape[0]

