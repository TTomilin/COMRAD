---
theme: seriph
layout: cover
background: https://wp.sfdcdigital.com/en-us/wp-content/uploads/sites/4/2025/08/multi-agent-systems-1.webp?resize=768,417
transition: slide-left
---

<div class="absolute inset-0 bg-black/30"/>
  <div class="relative z-10">
  <h1 class="text-8xl font-black tracking-tight text-white">COMRAD</h1>
  <h3 class="text-2xl mt-4 text-white/80 font-light">Cooperative Benchmark for Multi-Agent Reinforcement Learning in Doom</h3>
</div>

---
layout: image-right
image: https://datalab.flitto.com/en/company/blog/wp-content/uploads/OpenAI-1X-AI-Humanoid-Eve-Robot-Lifting-Box.jpg
transition: slide-up
---

# Single-Agent AI
*Like moving a box by yourself.*

+ **Simple:** You control everything.
+ **Omniscient:** You see the box, the door, and the path.
+ **Independent:** No need to coordinate.

---
layout: image-left
image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSQAYrXo46YBGf_Ak-tVXQmkSRQXxrmx-JxXQ&s
---

# Multi-Agent AI (The Reality)
*Like moving a heavy couch up a stair.*

+ **Purely Cooperative:** Individual strength isn't enough.
+ **Partially Observable:** You only see your side of the couch.
+ **Egocentric View:** You must communicate to know what your partner sees.
+ **Costly mistakes:** Collisions mean dropping the couch -> Huge cost.

---
layout: center
---

# The Problem: We train on "Flat Floors"

Most MARL benchmarks today (Overcooked, Gridworlds, SMAC):
+ Top-down or 2D views.
+ Simple physics and dynamics.
+ *Like practicing couch-moving on a flat, empty floor.*

**There is a huge gap for high-dimensional, partially observable, egocentric benchmarks.**

---
layout: two-cols
---

# Why Game-Based RL?

+ **Safe Sandbox:** Simulates complex physics and dynamics without real-world risks or hardware damage.
+ **The Stepping Stone:** Bridges the gap to real-world robotics.

<br>
<br>

# Why Doom?

+ **Egocentric & 3D:** True first-person view.
+ **High-Dimensional:** Complex pixel-level observations.
+ **Highly Throughput**

::right::

<div class="flex justify-center items-center h-full">
  <img src="https://upload.wikimedia.org/wikipedia/en/5/57/Doom_cover_art.jpg" class="w-2/3 rounded shadow-lg" />
</div>

---
layout: image-right
image: https://img.freepik.com/premium-vector/businessman-jump-cliff-mountain-gap_38887-123.jpg?semt=ais_hybrid&w=740&q=80
transition: fade
---

# The Gap in Doom MARL

*Wait, doesn't Doom MARL already exist?* Yes, but...

+ **Mostly Competitive:** Heavily focused on deathmatches (shooting each other).
+ **Ad-hoc Tasks:** Simple scenarios like "Defend the Center".
+ **No Standards:** There are no widely adopted benchmarks for **purely cooperative** MARL in 3D spaces.

---
layout: image-right
image: https://images.unsplash.com/photo-1518770660439-4636190af475?w=800
---

# The Compute Bottleneck

To learn to move the couch perfectly, agents need to fail *millions* of times.

+ Most existing 3D/realistic benchmarks are **slow**.
+ Slow simulation = Expensive compute.
+ Slow iterations = Stalled research progress.

---
layout: center
---

# Introducing COMRAD
The "Standardized Staircase" for MARL

We fill the 3 major gaps in current research:

1. **Egocentric & 3D:** Shifting the landscape away from flat SMAC/Gridworlds.
2. **Purely Cooperative:** Providing standard teamwork tasks and baseline algorithms.
3. **Blazing Fast (> 50,000 FPS):** Start your run in the early morning, and by breakfast, you have a well-converged neural network!

---
layout: center
class: text-center
---

# COMRAD
Moving MARL off the flat floor and up the stair.
