import numpy as np
from hmmlearn import hmm

states = ["-Hot-", "-Cold-"]
#array index:  0,   1 

observations = ["1", "2", "3"]
#array index:   0, 1, 2

model = hmm.CategoricalHMM(n_components=2)
model.startprob_ = np.array([0.8, 0.2])
model.transmat_ = np.array([[0.7, 0.3], 
                            [0.4, 0.6]])
model.emissionprob_ = np.array([[0.2, 0.4, 0.4], [0.5, 0.4, 0.1]])

obs_sqn_1 = np.atleast_2d([2, 0, 1, 2, 0, 1, 2, 0, 1]).T
# print(model.decode(obs_sqn_1))
logprob, decode_states_1 = model.decode(obs_sqn_1, algorithm="viterbi")
print ("Seq 1 decoded states: " + "".join(map(lambda x: states[x], decode_states_1)))

obs_sqn_2 = np.atleast_2d([2, 0, 0, 1, 2, 2, 0, 0, 1]).T
logprob, decode_states_2 = model.decode(obs_sqn_2, algorithm="viterbi")
print ("Seq 2 decoded states: " + "".join(map(lambda x: states[x], decode_states_2)))

input("Press Enter to continue...")
