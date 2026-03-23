---
theme: seriph
layout: cover
background: https://wp.sfdcdigital.com/en-us/wp-content/uploads/sites/4/2025/08/multi-agent-systems-1.webp?resize=768,417
transition: slide-left
themeConfig:
  primary: '#347563'
---

# COMRAD

### Cooperative Benchmark for Multi-agent Reinforcement Learning in Doom

---
layout: two-cols
---

# Single-Agent AI
Like moving a box by yourself.

+ Simple to measure
+ Independent. No need to coordinate.

::right::

# Multi-Agent AI
Like moving a heavy couch up winding stairs.

+ Individual strength isn't enough.
+ Coordination is everything. (Who lifts? Who backs up?)
+ Collisions are costly.

---
layout: center
---

# The Evaluation Problem
## How do we know which team of movers is the *best*?


---

# The Solution: A Benchmark
The "Standardized Staircase"

Instead of watching teams move random furniture in random houses, we build a **standardized obstacle course**:

+ Same dimensions
+ Same couch weight
+ Same sharp corners
+ **Universal scoring system**
