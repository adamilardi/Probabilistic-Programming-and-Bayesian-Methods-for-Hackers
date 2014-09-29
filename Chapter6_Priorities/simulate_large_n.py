import matplotlib
# matplotlib.use('Agg')
import numpy as np
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import scipy.stats as stats
from other_strats import *
from sets import Set


number_of_bandits = 500
max_pay_out_rate = .0065
hidden_prob = np.random.rand(number_of_bandits)
hidden_prob = (((hidden_prob - 0) * max_pay_out_rate) / 1) + 0.00001


top_indexes = sorted(range(len(hidden_prob)), key=lambda k: hidden_prob[k])
print 'Number of bandits %s and max payout %s' % (number_of_bandits, max_pay_out_rate)
print 'top 10 bandit indices: %s' % top_indexes[-10:]
print 'top 10 hidden probabilities: %s' % hidden_prob[top_indexes[-10:]]
print 'worse bandits are %s' % hidden_prob[top_indexes[:10]]
assert(hidden_prob[top_indexes[-1]] == hidden_prob.max())

bandits = Bandits(hidden_prob)


print 'The best bandit is %s in index %s' % (hidden_prob.max(), top_indexes[-1])

def regret(probabilities, choices):
    w_opt = probabilities.max()
    return (w_opt - probabilities[choices.astype(int)]).cumsum()


strategies = [bayesian_bandit_choice]
algos = []
for strat in strategies:
    algos.append(GeneralBanditStrat(bandits, strat))


def show_stats(strat):
    # _regret = regret(hidden_prob, strat.choices)
    top_bandits = sorted(range(len(hidden_prob)), key=lambda k: strat.trials[k])
    intersect = Set(top_bandits[-10:]).intersection(Set(top_indexes[-10:]))
    # print 'How many bandits did we find in the top ten: %s. They are %s and their hidden probablities are %s' % (len(intersect), intersect, hidden_prob[[x for x in intersect]])
    print 'Hidden probablities of the top 10 bandits %s' % hidden_prob[top_bandits[-10:]]
    # print 'number of pulls of top 10 bandits: %s' % strat.trials[top_bandits[-10:]]
    print 'How many bandits did we find in the top ten: %s.' % len(intersect)
    print 'Did we find the best bandit: %s' % (hidden_prob.max() == hidden_prob[top_bandits[-1]])
    print '#####'
    # print 'total regret %s' % _regret.sum()

def trial_incrementor(number_of_bandits):
    val = number_of_bandits
    while(True):
        val = val*10
        yield val

for strat in algos:
    trial_incrementor = trial_incrementor(number_of_bandits)
    already_run_trials = 0
    num_trails = trial_incrementor.next()
    for idx in range(3):
        trials_to_run = num_trails - already_run_trials
        print 'Running %s more trials' % trials_to_run
        strat.sample_bandits(trials_to_run)
        already_run_trials += trials_to_run
        num_trails = trial_incrementor.next()
        show_stats(strat)

    print 'Total trials %s' % already_run_trials

