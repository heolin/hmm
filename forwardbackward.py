#!/usr/bin/env python2
# -*- coding:utf-8 -*-
import numpy as np
from hmm import HMM

class ForwardAlgorithm(object):
    """Wynikiem pracy algorytmu Forward jest prawdopodobieństwo
       ciągu obserwacji i ostatniego stanu w zadanym modelu HMM"""
    def __init__(self, hmm):
        self.hmm = hmm

    def forward(self, emissions):
        alphas = self.forward_step(emissions)
        return self.get_probability(alphas)

    def get_probability(self, alphas):
        return sum(alphas[-1])

    def forward_step(self, emissions):
        # w tym kodzie Y to stany, X to obserwacje
        alphas = []
        alpha = np.zeros(self.hmm.states_count)
        for y in xrange(self.hmm.states_count):
            alpha[y] = self.hmm.initial_dist[y] * self.hmm.emission_probs[y][emissions[0]]
        alphas.append(alpha)

        for t in xrange(1, len(emissions)):
            alpha = np.zeros(self.hmm.states_count)
            for y in xrange(self.hmm.states_count):
                for y0 in xrange(self.hmm.states_count):
                    alpha[y] += alphas[t-1][y0] * self.hmm.transition_probs[y0][y]
                alpha[y] *= self.hmm.emission_probs[y][emissions[t]]
            alphas.append(alpha)
        return np.row_stack(alphas)



class BackwardAlgorithm(object):
    """Prawdopodobieństwo sekwencji obserwacji
       przy założeniu ostatniego stanu"""
    def __init__(self, hmm):
        self.hmm = hmm

    def backward(self, emissions):
        betas = self.backward_step(emissions)
        return self.get_probability(betas)

    def get_probability(self, betas, emissions):
        prob = 0
        for y in xrange(self.hmm.states_count):
            prob += self.hmm.initial_dist[y] * self.hmm.emission_probs[y][emissions[0]] * betas[0][y]
        return prob


    def backward_step(self, emissions):
        T = len(emissions)

        betas = []
        for t in xrange(T):
            betas.append(np.zeros(self.hmm.states_count))

        for y in xrange(self.hmm.states_count):
            betas[T-1][y] = 1

        for t in reversed(range(T-1)):

            for y in xrange(self.hmm.states_count):
                for y1 in xrange(self.hmm.states_count):
                    betas[t][y] += betas[t+1][y1] * self.hmm.transition_probs[y][y1] * self.hmm.emission_probs[y1][emissions[t+1]]
        return np.row_stack(betas)



class BaumWelchAlgorithm(object):
    """Prawdopodobieństwo stanu przy założeniu sekwencji obserwacji"""
    def __init__(self, hmm):
        self.hmm = hmm
        self.forward_algorithm = ForwardAlgorithm(hmm)
        self.backward_algorithm = BackwardAlgorithm(hmm)

    def update_step(self, emissions):
        T = len(emissions)
        alphas = self.forward_algorithm.forward_step(emissions)
        betas = self.backward_algorithm.backward_step(emissions)
        gammas = self.get_gammas(alphas, betas, emissions, T)
        deltas = self.get_deltas(alphas, gammas, T)
        return alphas,  betas, gammas, deltas

    def get_gammas(self, alphas, betas, emissions, T):
        prh = self.forward_algorithm.get_probability(alphas)

        gammas = np.zeros((len(alphas)-1, self.hmm.states_count, self.hmm.states_count))

        for t in xrange(T-1):
            for s in xrange(self.hmm.states_count):#t
                for u in xrange(self.hmm.states_count):#s
                    gammas[t, s, u] = alphas[t, s] * self.hmm.transition_probs[s, u] * self.hmm.emission_probs[u][emissions[s+1]] * betas[t+1, u]
                    gammas[t, s, u] /= prh
        return gammas


    def get_deltas(self, alphas, gammas, T):
        deltas = np.zeros((T, self.hmm.states_count))

        for t in xrange(T):
            deltas[t] = np.array(alphas[t]/sum(alphas[t]))

        return deltas


    def get_initial_probs(self, values, corpus):
        new_initial_prob = np.zeros(self.hmm.states_count)
        for s in xrange(self.hmm.states_count):
            prob = 0
            for emissions, counts in corpus:
                prob += values[emissions][3][0][s] * counts
            new_initial_prob[s] = prob
        new_initial_prob /= sum(new_initial_prob)
        return new_initial_prob


    def get_emission_probs(self, values, corpus):
        new_emission_probs = np.zeros((self.hmm.states_count, self.hmm.states_count))
        for s in xrange(self.hmm.states_count): # dla każdego stanu
            for a in xrange(self.hmm.emission_count): #dla każdego znaku z alfabetu
                prob = 0
                for emissions, counts in corpus:
                    for t in xrange(len(emissions)):
                        if emissions[t] == a:
                            prob += values[emissions][3][t][s] * counts

                new_emission_probs[s, a] = prob
            new_emission_probs[s] /= sum(new_emission_probs[s])
        return new_emission_probs

    def get_transition_probs(self, values, corpus):
        new_transition_probs = np.zeros((self.hmm.states_count, self.hmm.states_count))
        for s in xrange(self.hmm.states_count):
            for u in xrange(self.hmm.states_count):
                prob = 0
                for emissions, counts in corpus:
                    for t in xrange(len(emissions)-1):
                        prob += values[emissions][2][t][s][u] * counts
                new_transition_probs[s, u] = prob
            new_transition_probs[s] /= sum(new_transition_probs[s])
        return new_transition_probs


    def update(self, corpus):
        values = {}
        for emissions, _ in corpus:
            values[emissions] = self.update_step(emissions)

        new_initial_prob = self.get_initial_probs(values, corpus)
        new_transition_probs = self.get_transition_probs(values, corpus)
        new_emission_probs = self.get_emission_probs(values, corpus)

        return new_initial_prob, new_transition_probs, new_emission_probs


def main():
    transition_probs = np.array([[0.3, 0.7], [0.1, 0.9]])
    emission_probs = np.array([[0.4, 0.6], [0.5, 0.5]])
    initial_dist = np.array([0.85, 0.15])

    hmm = HMM(initial_dist, transition_probs, emission_probs)

    corpus = [((0, 1, 1, 0), 10), ((1, 0, 1), 20)]

    print sum(ForwardAlgorithm(hmm).forward(emissions) for emissions, count in corpus)

    for _ in xrange(100):
        baum_welch = BaumWelchAlgorithm(hmm)
        initial_dist, transition_probs, emission_probs = baum_welch.update(corpus)
        hmm = HMM(initial_dist, transition_probs, emission_probs)
        print sum(ForwardAlgorithm(hmm).forward(emissions) for emissions, count in corpus)

if __name__ == "__main__":
    main()
