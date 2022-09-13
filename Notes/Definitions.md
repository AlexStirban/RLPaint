# Definitions

## MDP - Markov Decision Process
Mathematical framework for modeling decision making in stochastic situations that are partly under the control of a decision maker. They are considered an extension of Markov chains.

## Model-based RL
In model-based RL we use predictions of the environment's response without actually
needing to interact always with it.

## Model-free RL
In model-free we refine a policy based on the agent's interactions with the environment.



## Value functions
Represents "how good is it to perform a given action in a given state" or "how good is for the agent to be in agiven state", where "how good" is defined in terms of the future rewards that could be expected given a set of actions, thus depending on a particular policy. 

If a policy $\pi $ is a mapping between each sate $s \in S$ and $a \in A(s)$ to the probability $\pi(s, a)$ of taking action $a$ under the state $s$; then the value of a state $s$ under a policy $\pi$ is the expected return when starting at $s$ and following $\pi$.
For MDPs, formally the state-value function is defined as:
$$
    V^{\pi}(s) = E_{\pi}\left\{ R_t | s_t = s \right\} = E_\pi \left\{ \sum^{\infty}_k \gamma^kr_{t+k+1} \bigg\lvert s_t = s\right\}

$$

Simalarly, the value of taking an action $a$ in a state $s$ is defined as $Q^{\pi}(s, a)$ under a policy $\pi$ is:

$$
    Q^{\pi}(s, a) = E_{\pi}\left\{ R_t | s_t = s, a_t = a \right\} = E_\pi \left\{ \sum^{\infty}_k \gamma^kr_{t+k+1} \bigg\lvert s_t = s, a_t = a\right\}
$$