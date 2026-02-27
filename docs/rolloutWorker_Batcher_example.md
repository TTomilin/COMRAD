## Why this here
This gives example of how signal-slot system works, kind of similar to TCP. Refer to docs for more details.

### Before processes start running
1: Runner creates components:
- RolloutWorker (simulates environments)
- Batcher (collects trajectories for training)

2: Each component defines slots :
- RolloutWorker has a method that can emit a signal called `"p0_trajectories"`
- Batcher has a slot method called `on_new_trajectories()`

3: Runner connects them:
```python
rollout_worker.signal("p0_trajectories").connect(batcher.on_new_trajectories)
```
This creates a direct connection, or "wire" from that specific signal to that specific slot.

### When processes running

4: RolloutWorker finishes collecting a trajectory and emits:
```python
self.emit_signal("p0_trajectories", trajectory_data)
```
5: The signal-slot system atuo routes this to the Batcher's `on_new_trajectories()` method, which gets called with `trajectory_data` as an argument.
6: Batcher's `on_new_trajectories()` method executes and processes the trajectory.
