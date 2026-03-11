## Agent count FPS scaling
+ FPS is per agent, divide by `N * frameskip` for env steps/sec
+ For example with frameskip being 4:
    + N=2: 9,825 FPS -> 1,228 steps/s
    + N=3: 10,637 FPS -> 886 steps/s
    + With more agent `env_step` profile increases 11.5% (285s -> 318s), inference not changes
