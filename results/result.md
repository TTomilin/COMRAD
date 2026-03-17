## Nodes

| ... | mcs.gpu.q | tue.gpu.q |
|---|---|---|
| Node | mcs-gpub001 | tue-gpub001 |
| GPU | NVIDIA GeForce RTX 2080 Ti | NVIDIA L4 |
| GPU Architecture | Turing (TU102) | Ada Lovelace (AD104) |
| GPU FP32 TFLOPS | 13.4 | 30.3 |
| GPU Memory | 11 GB GDDR6 | 24 GB GDDR6 |
| Mem bandwidth | 600 | 300 |

| ... | RTX 2080 Ti | L4 |
|---|---|---|
| FPS | 14,424 | 18,441 |
| Peak 300s-avg FPS | 15,079 | 19,550 |
| Total frames collected | 161,701,888 | 206,897,152 |
| GPU compute utilization | 75.1% avg, max 79% | 71.5% avg, max 98% |
| GPU memory utilization | 11.2% avg, max 12% | 21.3% avg, max 65% |
| GPU memory allocated | 2.9 GB / 11 GB (26%) | 3.1 GB / 24 GB (13%) |

### Profile tree (pasted from excel)

| ... | mcs-gpub001 | tue-gpub001 | ratio |
|---|---|---|---|
| handle_policy_step | 19,953 | 18,477 | -7.4% |
| obs_to_device_normalize | 5,126 | 5,127 | +0.0% |
| forward | 9,276 | 7,760 | -16.3% |
| epsilon_greedy | 2,738 | 3,099 | +13.2% |
| prepare_outputs | 1,774 | 1,577 | -11.1% |
| send_messages | 309 | 297 | -3.9% | 1.6% |
| deserialize | 48 | 42 | -12.5% |
| stack | 110 | 77 | -30.0% |
| env_step (w0) | 9,075 | 9,269 | +2.1% |
| env_step (w3) | 9,085 | 9,099 | +0.2% |
| wait_policy_total (iw0) | 772 | 1,633 | +111.7% |
| wait_policy_total (iw1) | 853 | 1,551 | +81.8% |
| train (learner) | 4,032 | 3,699 | -8.3% |
| add_to_buffer | 1,992 | 1,440 | -27.7% |
| batching (batcher) | 562 | 354 | -37.0% |
| enqueue_policy_requests (w0) | 395 | 287 | -27.3% |
| process_policy_outputs (w0) | 173 | 100 | -42.2% |

### Conlcusion

+ GPU is bottleneck for inference `forward`. Because of this:
    + On L4 `wait_policy_total` is longer as L4 is faster -> process faster and wait longer
    + batcher `batching` spends less time assembling batches because inference completes faster
    + `add_to_buffer` decreases becus faster CPU-GPU (I guess)
    + `enqueue/process_policy` decreases as less contention on shared memory queues when inference drains requests faster

+ On faster GPU, bottleneck comes from CPU env collection
    + On L4, GPU is powerful -> CPU is bottleneck
    + On RTX, GPU is closer to a bottleneck



## Agent count FPS scaling
+ FPS is per agent, divide by `N * frameskip` for env steps/sec: `env_steps_per_sec = FPS / (N * frameskip)`
+ For example with frameskip being 4:
    + N=2: 9,825 FPS -> 1,228 steps/s
    + N=3: 10,637 FPS -> 886 steps/s
    + With more agent `env_step` profile increases 11.5% (285s -> 318s), inference not changes

### HAPPO on mcs-gpub001

```
--algo=HAPPO --train_for_env_steps=5000000 --num_workers=4 --num_envs_per_worker=4
--policy_workers_per_policy=2 --batch_size=2048 --env_frameskip=4 --use_rnn=True
--happo_critic_rnn=True
```

| ... | N=2 | N=3 | ratio |
|---|---|---|---|
| FPS | 9,824.8 | 10,637.3 | +8.3% |
| Per-agent FPS (FPS / N) | 4,912.4 | 3,545.8 | -27.8% |
| env_steps_per_sec (FPS / N / frameskip) | 1,228.1 | 886.4 | -27.8% |
| main_loop | 511.1s | 472.2s | -7.6% |
| Total frames | 5,021,696 | 5,022,720 | |
| Time | 511 | 472.2 | |

+ FPS incr as each environment step produces more frames (more agents),
+ N=2: env steps = `5,021,696 / (2*4) = 627,712` (627,712 env steps / 1,228.1 env_steps_per_sec = 511.0s)
+ N=3: env steps = `5,022,720 / (3*4) = 418,560`

-> N=3 executes 33% fewer actual environment steps to reach the same frame count, so it finishes sooner

| ... | N=2 | N=3 | ratio |
|---|---|---|---|
| env_step (w0) | 285.1 | 317.9 | +11.5% |
| env_step (w3) | 285.0 | 317.7 | +11.5% |
| forward (iw0) | 277.3 | 270.2 | -2.6% |
| obs_to_device (iw0) | 73.7 | 60.9 | -17.4% |
| prepare_outputs (iw0) | 40.5 | 30.0 | -26.0% |
| train (learner) | 462.8 | 414.8 | -10.3% |
| wait_policy_total (iw0) | 28.3 | 31.6 | +11.7% |
| send_messages (iw0) | 4.4 | 3.4 | -22.7% |
| batching | 21.0 | 22.1 | +5.2% |
| enqueue_policy_requests (w0) | 13.0 | 10.0 | -23.1% |

+ For N=3, there are 3 UDP send per tic instead of 2. Over 4 frameskip: 12 vs 8 UDP send per step -> `env_step` increases
+ For N=3, more obs per inference request so `prepare_outputs` and `obs_to_device` decreases

-> N=3 is more efficient in time because of larger inference batch size

### QMIX on tue-gpub001 L4
```
--algo=QMIX --mixer=qmix --train_for_env_steps=10000000 --num_workers=4 --num_envs_per_worker=4
--policy_workers_per_policy=2 --batch_size=1920 --env_frameskip=4 --use_rnn=True
--dqn_max_updates_per_batch=1 --train_frequency=2 --batched_sampling=True
```

| ... | N=2 | N=3 | ratio |
|---|---|---|---|
| FPS | 16,719 | 14,481 | -13.4% |
| Per-agent FPS (FPS / N) | 8,360 | 4,827 | -42.3% |
| env_steps_per_sec (FPS / N / frameskip) | 2,090 | 1,207 | -42.3% |
| main_loop | 599.0s | 691.6s | +15.4% |
| Total frames | 10,014,720 | 10,014,720 | |

+ For QMIX it actually runs longer for 3 agents as env_steps_per_sec is longer (each step is slower)
+ `wait_policy_total` increases around 186% as env collection is bottleneck
+ `forward` devrease as less inference calls because fewer env steps
+ `train` increases 50.8%

### Conclusion

+ The throughput loss from adding agents is worse on faster hardware
    + Faster L4 GPU makes `env_step` (CPU) bottleneck -> Becomes throughput loss
    + On the slower 2080 Ti, inference `forward` time covers the time of CPU env collection `env_step`

## So what next

+ In all cases, on faster GPU, with more agents GPU is not utilized as much because env collection is not fast enough
+ On fast GPU, maybe request 32/64 CPUs instead of 16 and increase rollout_workers, and increase inference batch size `num_envs_per_worker` as well
