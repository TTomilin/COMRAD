## Observation
Each `.step()` gives raw observations, for example:
```
{'obs': tensor([[[[107,  99, 107,  ..., 119, 119, 127], ...
[ 19,  11,  47,  ...,  47,  67,  55]]]], dtype=torch.uint8)}
torch.Size([2, 3, 72, 128])
```
Or if there is `extra_wrapper` with extra stuff like ammos, etc.:
```
{'obs': array([[[ 83, 127, 119, ..., 107,  79,  91], ...
[ 19,  11,  47, ...,  47,  67,  55]]], dtype=uint8),
'measurements': array([...], dtype=float32)}
```

This then goes to `Encoder` (feature extractor), to get feature tensor, the feature tensor then goes through `Core` (RNN) (either LSTM or GRU) to get a tensor of array of this state + previous state data and "highlights" feature. This tensor then gets passed to the `Decoder`, but there are 2 types of `Decoder`:
+ Actor head: returns policy logits/probabilities (Decides what to do, probabilistically)
+ Critic head: returns Value estimate $V(s)$ (This actually is estimate of the **expected future total reward** starting from the current state, assuming the agent follows its policy)

This whole "decoding" process happens inside forward pass.

The process is Encoder -> Core -> Decoder -> ..... And the "......" is the process where it goes backward (called backpropagation) and update the weight.

## Order of computations:
Actor/critic head (decoder) -> Advantage -> Loss -> Backprop: {Gradient -> Weight update}

Forward pass:
+ Actor head -> policy logits/probabilities
+ Critic head -> Value estimate $V(s)$

Compute Advantage:
+ Advantage is computed before any loss and is required to form the actor loss
+ Compute how much better a specific action is compared to the average action in a given state, i.e. how surprisingly good or bad the outcome was
+ This "Advantage" function thing is just a concept, it can be implemented using many formulas, not tied to 1 fixed formula.
	+ Advantage can be calculated with: Monte Carlo Advantage, TD Error, GAE, n-step average, etc.
	+ In this project, we use vtrace for off-policy and GAE for on-policy
+ $A=R+γV(s_{t+1})−V(s)$ (or $A=GAE(...)$), here $R$ is the **actual return**
+ $s_{t+1}$ is next state after taking action $a$, while $s$ it current state


Compute losses:
+ Actor: $L_{actor}​=−log\pi(a∣s) \cdot A$
+ Critic: $L_\text{critic} = (R + \gamma V(s_{t+1}) - V(s))^2$
+ Total loss: $L = L_\text{actor} + c \cdot L_\text{critic} - \beta H$
+ Compute total loss also auto merges gradients in shared actor critic network
	+ `L_total = L_actor + c * L_critic - B * H; L_total.backward(); optimizer.step()`
+ Another way that is less common is to backprop twice: backprop actor loss, keep the graph, then backprop critic loss, gradients accumulate.
	+ `actor_loss.backward(retain_graph=True); critic_loss.backward(); optimizer.step()`
	+ Same result, here final gradient is $\nabla L_{\text{shared}} = \nabla L_{\text{actor}} + c \nabla L_{\text{critic}}$
	+ Uses when actor and critic optimizers differ or for debugging


Backprop:
+ Compute gradients, then weight updates
