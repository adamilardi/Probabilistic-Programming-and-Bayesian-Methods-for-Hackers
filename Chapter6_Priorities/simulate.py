import matplotlib
import numpy as np
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import scipy.stats as stats
from other_strats import *
from sets import Set


number_of_bandits = 500
max_pay_out_rate = .03
hidden_prob = np.random.rand(number_of_bandits)
hidden_prob = (((hidden_prob - 0) * max_pay_out_rate) / 1) + 0.00001


top_indexes = sorted(range(len(hidden_prob)), key=lambda k: hidden_prob[k])
print 'top 10 bandit indices: %s' % top_indexes[-10:]
print 'top 10 hidden probabilities: %s' % hidden_prob[top_indexes[-10:]]
print 'bottom 10 hidden probabilities: %s' % hidden_prob[top_indexes[:10]]
assert(hidden_prob[top_indexes[-1]] == hidden_prob.max())


bandits = Bandits(hidden_prob)
print 'The best bandit is %s in index %s' % (hidden_prob.max(), top_indexes[-1])

def regret(probabilities, choices):
    w_opt = probabilities.max()
    return (w_opt - probabilities[choices.astype(int)]).cumsum()

# create new strategies
strategies = [random_choice, bayesian_bandit_choice]
algos = []
for strat in strategies:
    algos.append(GeneralBanditStrat(bandits, strat))


for strat in algos:
    strat.sample_bandits(500000)


#test and plot
plt.figure(0, figsize=(25, 15))
regrets = []
for i, strat in enumerate(algos):
    _regret = regret(hidden_prob, strat.choices)
    regrets.append(_regret)
    plt.plot(_regret, label=strategies[i].__name__, lw=3)
    top_bandits = sorted(range(len(hidden_prob)), key=lambda k: strat.trials[k])
    intersect = Set(top_bandits[-10:]).intersection(Set(top_indexes[-10:]))
    print 'How many bandits did we find in the top ten: %s they are %s and their hidden probablities are %s' % (len(intersect), intersect, hidden_prob[[x for x in intersect]])
    print 'Hidden probablities of the top 10 bandits %s' % hidden_prob[top_bandits[-10:]]
    print 'number of pulls of top 10 bandits: %s' % strat.trials[top_bandits[-10:]]


plt.title("Total Regret of Bayesian Bandits Strategy vs. Random guessing. Number of bandits %s. Max payout %s%%." % (number_of_bandits, max_pay_out_rate*100))
plt.xlabel("Number of pulls")
plt.ylabel("Regret after $n$ pulls")
plt.legend(loc="upper left")
plt.savefig('bandit_%s_%s.png' % (number_of_bandits, max_pay_out_rate))

regret_percent = regrets[1]/regrets[0]
plt.figure(1, figsize=(25, 15))
plt.plot(regret_percent, label='regret percent', lw=3)
plt.title("Percent of Bayesian Bandits Strategy regret out of Random guessing regret")
plt.ylim(2)
plt.xlabel("Number of pulls")
plt.ylabel("Regret percent after $n$ pulls")
plt.legend(loc="upper left")
plt.savefig('bandit_regret_%s_%s.png' % (number_of_bandits, max_pay_out_rate))



