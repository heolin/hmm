#!/usr/bin/env python2
# -*- coding:utf-8 -*-
import numpy as np
from hmm import HMM

class ViterbiAlgorithm(object):
    """Algorytm dla zadanej sekwencji obserwacji oblicza najbardziej
       prawdopodobną sekwencję stanów i jej prawdopodobieństwo."""
    def __init__(self, hmm):
        self.hmm = hmm
    def decode(self, emissions):
        T = len(emissions)
        V = np.zeros((T, self.hmm.states_count))
        path = {}

        for k in xrange(self.hmm.states_count):
            yt = emissions[0]
            V[0, k] = self.hmm.initial_dist[k] * self.hmm.emission_probs[k, yt]
            #ustawia initial probabilities w oparciu o emisje i początkowy rozkład
            path[k] = [k]

        for t in range(1, T): # dla każdego kolejnego elementu emisji
            yt = emissions[t]
            next_path = {}
            for k in xrange(self.hmm.states_count): #sprawdza wszystkie stany
                (prob, state) = max((self.hmm.emission_probs[k, yt] * self.hmm.transition_probs[x, k] * V[t-1, x], x) for x in xrange(self.hmm.states_count))
                #i wybiera poprzedni stan, o największym prawdopodobieństwie:
                #emisji danej obserwacji,
                #prawdopodobieństwie przejscia do obecnego stanu
                #wartości algorytmu obliczonej w poprzednim kroku
                V[t, k] = prob
                next_path[k] = path[state] + [k]
                #ścieżka obecnego stanu to ściażka wybranego przed chwilą
                #poprzedniego stanu rozszerzona o obecny
            path = next_path

        (prob, state) = max((V[T-1, y], y) for y in xrange(self.hmm.states_count))
        return prob, path[state]


def main():
    transition_probs = np.array([[0.7, 0.3], [0.4, 0.6]])
    emission_probs = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    initial_dist = np.array([0.6, 0.4])

    hmm = HMM(initial_dist, transition_probs, emission_probs)

    emissions = (0, 1, 2)
    print ViterbiAlgorithm(hmm).decode(emissions)


if __name__ == "__main__":
    main()
