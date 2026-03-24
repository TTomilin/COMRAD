---
theme: default
transition: fade
class: text-center
drawings:
  persist: false
css: unocss
---

<style>
:root {
  --slidev-theme-primary: #1a1a1a;
  --slidev-theme-accent: #e63946;
}

* {
  font-family: 'Helvetica Neue', 'Archivo', system-ui, -apple-system, sans-serif;
}

h1 {
  font-weight: 700;
  letter-spacing: -0.02em;
}

h2 {
  font-weight: 500;
  color: #555;
}

.small-text {
  font-size: 0.85rem;
}

.category-card {
  padding-left: 1rem;
  margin-bottom: 1rem;
}
</style>

<div class="h-full flex flex-col justify-center items-center">
  <h1 class="text-8xl font-black tracking-tight text-[#1a1a1a] mb-6">COMRAD</h1>
  <p class="text-2xl text-[#666] tracking-wide">Cooperative Multi-Agent Reinforcement Learning in Doom</p>
  <p class="text-lg text-[#999] mt-4">A standardized benchmark for 3D cooperative AI</p>
</div>

---

<div class="h-full flex flex-col justify-center px-16">
  <h1 class="text-5xl font-bold text-[#1a1a1a] mb-12">The Coordination Problem</h1>

  <div class="grid grid-cols-2 gap-16 items-center">
    <div class="border-l-4 border-[#e63946] pl-8">
      <p class="text-2xl text-[#333] leading-relaxed mb-6">
        <span class="font-semibold">Single-agent:</span><br/>
        You control everything. You see everything.
      </p>
    </div>
    <div class="border-l-4 border-[#1a1a1a] pl-8">
      <p class="text-2xl text-[#333] leading-relaxed mb-6">
        <span class="font-semibold">Multi-agent:</span><br/>
        Partial view. Must coordinate. Mistakes compound.
      </p>
    </div>
  </div>

  <p class="text-xl text-[#888] mt-6 text-center italic">
    Like moving a couch up stairs. Each person only sees their side.
  </p>
  <div class="grid grid-cols-2 gap-8 mt-4" style="height: 60%">
    <div class="flex flex-col items-center gap-2">
      <div style="width:auto;border-radius:8px; border: 2px solid #1a1a1a;">
        <img src="https://datalab.flitto.com/en/company/blog/wp-content/uploads/OpenAI-1X-AI-Humanoid-Eve-Robot-Lifting-Box.jpg"
             style="width:auto;height:auto;display:block;" />
      </div>
      <p class="text-sm text-[#666]">Single agent: carries a box alone</p>
    </div>
    <div class="flex flex-col items-center gap-2">
      <div style="height:67%; border-radius:8px; border: 2px solid #1a1a1a; overflow: hidden;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSQAYrXo46YBGf_Ak-tVXQmkSRQXxrmx-JxXQ&s"
             style="width:auto;height:auto;display:block;" />
      </div>
      <p class="text-sm text-[#666]">Multi-agent: moving a couch up stairs</p>
    </div>
  </div>
</div>

---

<div class="h-full flex flex-col justify-center px-16">
  <h1 class="text-5xl font-bold text-[#1a1a1a] mb-12">The Gap</h1>

  <p class="text-2xl text-[#555] mb-10">Current multi-agent benchmarks use simplified 2D/3D worlds.</p>

  <div class="grid grid-cols-2 gap-8 text-center">
    <div class="p-6 bg-[#fafafa] border border-[#ddd] rounded-lg">
      <p class="text-lg font-semibold text-[#666] mb-2">StarCraft (SMAC)</p>
      <p class="text-sm text-[#999]">Top-down view, micromanagement</p>
    </div>
    <div class="p-6 bg-[#fafafa] border border-[#ddd] rounded-lg">
      <p class="text-lg font-semibold text-[#666] mb-2">Google Research Football (GRF)</p>
      <p class="text-sm text-[#999]">Physics-based 3D sports sim</p>
    </div>
    <div class="p-6 bg-[#fafafa] border border-[#ddd] rounded-lg">
      <p class="text-lg font-semibold text-[#666] mb-2">Megaverse</p>
      <p class="text-sm text-[#999]">High-speed 3D training (mainly single agent)</p>
    </div>
    <div class="p-6 bg-[#fafafa] border border-[#ddd] rounded-lg">
      <p class="text-lg font-semibold text-[#666] mb-2">Overcooked / Gridworlds</p>
      <p class="text-sm text-[#999]">Simple discrete coordination</p>
    </div>
  </div>

  <p class="text-xl text-[#e63946] mt-10 text-center font-medium">
    No standardized benchmark for 3D first-person cooperative AI exists.
  </p>
</div>

---

<div class="h-full flex flex-col justify-center px-16">
  <h1 class="text-5xl font-bold text-[#1a1a1a] mb-12">COMRAD</h1>

  <p class="text-sm font-medium tracking-widest uppercase text-[#88888] mt-12">
    Moving MARL off the flat floor and up the stair.
  </p>

  <div class="grid grid-cols-3 gap-10">
    <div class="border-t-4 border-[#e63946] pt-6">
      <h3 class="text-2xl font-bold text-[#1a1a1a] mb-3">First-Person 3D</h3>
      <p class="text-[#666]">Agents see what humans see: partial, egocentric observations</p>
    </div>
    <div class="border-t-4 border-[#e63946] pt-6">
      <h3 class="text-2xl font-bold text-[#1a1a1a] mb-3">Standardized</h3>
      <p class="text-[#666]">Reproducible scenarios with baseline algorithms</p>
    </div>
    <div class="border-t-4 border-[#e63946] pt-6">
      <h3 class="text-2xl font-bold text-[#1a1a1a] mb-3">High Throughput</h3>
      <p class="text-[#666]">>50,000 FPS -> rapid iteration and research progress</p>
    </div>
  </div>

  <p class="text-center text-lg text-[#888] mt-3">
    Bridging the gap between toy benchmarks and real-world robotics.
  </p>
</div>

---

<div class="h-full flex flex-col justify-center px-16">
  <h1 class="text-5xl font-bold text-[#1a1a1a] mb-8">Why It Matters</h1>

  <div class="grid grid-cols-3 gap-10 mb-6">
    <div class="text-center p-6 bg-[#fafafa] rounded-lg">
      <p class="text-5xl font-black text-[#e63946] mb-2">>50K</p>
      <p class="text-sm text-[#666]">Frames per second</p>
      <p class="text-xs text-[#999] mt-2">Train early morning, well-converged result by breakfast</p>
    </div>
    <div class="text-center p-6 bg-[#fafafa] rounded-lg">
      <p class="text-5xl font-black text-[#e63946] mb-2">13+</p>
      <p class="text-sm text-[#666]">Benchmark scenarios</p>
      <p class="text-xs text-[#999] mt-2">Reproducible, standardized tasks</p>
    </div>
    <div class="text-center p-6 bg-[#fafafa] rounded-lg">
      <p class="text-5xl font-black text-[#e63946] mb-2">7</p>
      <p class="text-sm text-[#666]">Baseline algorithms</p>
      <p class="text-xs text-[#999] mt-2">QMIX, MAPPO, HAPPO, + more</p>
    </div>
  </div>

  <div class="border border-[#eee] rounded-lg p-2 bg-[#f5f5f5] inline-flex items-center justify-center self-center">
    <img src="./scenarios.gif" alt="scenarios" class="rounded h-48" />
  </div>
</div>

---

<div class="h-full flex flex-col justify-center px-16">
  <h1 class="text-4xl font-bold text-[#1a1a1a]" style="margin-bottom: 3rem;">Scenario Design: Four Property Categories</h1>

  <div class="grid grid-cols-2 gap-8">
    <div class="category-card">
      <h3 class="text-xl font-bold text-[#1a1a1a] mb-2">Game-Theoretic Properties</h3>
      <div style="width: 5em; border-bottom: 2px solid #e63946; margin-bottom: 0.5rem;"></div>
      <p class="text-sm text-[#666]">Social dilemmas, coordination games, asymmetric payoffs, Nash equilibria testing</p>
    </div>
    <div class="category-card">
      <h3 class="text-xl font-bold text-[#1a1a1a] mb-2">Game Design Properties</h3>
      <div style="width: 5em; border-bottom: 2px solid #e63946; margin-bottom: 0.5rem;"></div>
      <p class="text-sm text-[#666]">Role asymmetry, spatial constraints, resource scarcity, temporal pressure</p>
    </div>
    <div class="category-card">
      <h3 class="text-xl font-bold text-[#1a1a1a] mb-2">Reinforcement Learning Properties</h3>
      <div style="width: 5em; border-bottom: 2px solid #e63946; margin-bottom: 0.5rem;"></div>
      <p class="text-sm text-[#666]">Partial observability, credit assignment, sparse rewards, exploration challenges</p>
    </div>
    <div class="category-card">
      <h3 class="text-xl font-bold text-[#1a1a1a] mb-2">Emergent Behavior Properties</h3>
      <div style="width: 5em; border-bottom: 2px solid #e63946; margin-bottom: 0.5rem;"></div>
      <p class="text-sm text-[#666]">Role specialization, implicit communication, altruism, trust formation</p>
    </div>
  </div>

  <p class="text-center text-sm text-[#888] mt-8 italic">
    Each scenario is designed to stress-test specific combinations of these properties.
  </p>
</div>

---

<div class="h-full flex flex-col justify-center px-16">
  <h1 class="text-4xl font-bold text-[#1a1a1a] mb-6">Scenario Examples</h1>

  <div class="grid grid-cols-3 gap-6 text-sm">
    <div class="p-4 border border-[#eee] rounded-lg">
      <h4 class="font-bold text-[#1a1a1a] mb-2">Armory Siege</h4>
      <p class="text-[#666] mb-2">Defend core + fetch ammo runs</p>
      <p class="text-xs text-[#999]">Role switching, temporal pressure</p>
    </div>
    <div class="p-4 border border-[#eee] rounded-lg">
      <h4 class="font-bold text-[#1a1a1a] mb-2">Stag Hunt Arena</h4>
      <p class="text-[#666] mb-2">Hunt stag together or hare alone</p>
      <p class="text-xs text-[#999]">Coordination dilemma, trust</p>
    </div>
    <div class="p-4 border border-[#eee] rounded-lg">
      <h4 class="font-bold text-[#1a1a1a] mb-2">Stealth Labyrinth</h4>
      <p class="text-[#666] mb-2">Torch-holder + Gunner in darkness</p>
      <p class="text-xs text-[#999]">Extreme Dec-POMDP, implicit comms</p>
    </div>
    <div class="p-4 border border-[#eee] rounded-lg">
      <h4 class="font-bold text-[#1a1a1a] mb-2">Common Harvest</h4>
      <p class="text-[#666] mb-2">Harvest vs. maintain shared resource</p>
      <p class="text-xs text-[#999]">Tragedy of the commons</p>
    </div>
    <div class="p-4 border border-[#eee] rounded-lg">
      <h4 class="font-bold text-[#1a1a1a] mb-2">Lava Maze</h4>
      <p class="text-[#666] mb-2">Navigator + Spectator guidance</p>
      <p class="text-xs text-[#999]">Asymmetric roles, partial obs</p>
    </div>
    <div class="p-4 border border-[#eee] rounded-lg">
      <h4 class="font-bold text-[#1a1a1a] mb-2">Rhythm Sync</h4>
      <p class="text-[#666] mb-2">Simultaneous switch pressing</p>
      <p class="text-xs text-[#999]">Temporal coordination, no comms</p>
    </div>
  </div>

  <p class="text-center text-sm text-[#888] mt-6">
    13+ scenarios covering social dilemmas, role asymmetry, and implicit coordination
  </p>
</div>

---

<div class="h-full flex flex-col justify-center px-16">
  <h1 class="text-4xl font-bold text-[#1a1a1a] mb-10" style="padding-bottom: 20px;">Procedural Generation: Non-Stationary Environments</h1>
  <div class="grid grid-cols-2 gap-12 items-start">
    <div>
      <h3 class="text-sm uppercase tracking-widest text-[#888] mb-3 border-b border-[#ddd] pb-2">Map Generation</h3>
      <div class="text-sm text-[#666] space-y-3">
        <p><span class="font-semibold text-[#1a1a1a]">Voronoi-based:</span> Relaxed Voronoi cells -> rooms/corridors</p>
        <p><span class="font-semibold text-[#1a1a1a]">Seedable:</span> Fully deterministic for reproducible experiments</p>
        <p><span class="font-semibold text-[#1a1a1a]">Batch Generation:</span> Parameter sweeps create 100+ layout variants</p>
      </div>
    </div>
    <div>
      <h3 class="text-sm uppercase tracking-widest text-[#888] mb-3 border-b border-[#ddd] pb-2">Prevent Overfitting</h3>
      <div class="text-sm text-[#666] space-y-3">
        <p><span class="font-semibold text-[#1a1a1a]">Layout Variation:</span> Different maze structures per seed</p>
        <p><span class="font-semibold text-[#1a1a1a]">Runtime Reseeding:</span> Environment shifts mid-episode</p>
        <p><span class="font-semibold text-[#1a1a1a]">Parameter Grids:</span> Distance, difficulty, timing all vary</p>
      </div>
    </div>
  </div>

  <div class="p-4 bg-[#fafafa] rounded-lg">
    <p class="text-sm text-[#666] mb-2"><span class="font-semibold">Example: Armory Siege batch parameters:</span></p>
    <p class="text-xs text-[#888] font-mono">distance: [600, 800, 1100] × corridor_width: [2, 4] × door_timer: [100, 350, 600] × ...</p>
    <p class="text-sm text-[#e63946] mt-2">= <span class="font-bold">162</span> distinct environment configurations</p>
  </div>
</div>

---

<div class="h-full flex flex-col justify-center px-16">
  <h1 class="text-4xl font-bold text-[#1a1a1a] mb-6">Baseline Algorithms</h1>

  <div class="grid grid-cols-3 gap-6">
    <div>
      <h3 class="text-lg font-semibold text-[#888] mb-3">Value Decomposition</h3>
      <div class="space-y-2">
        <div class="p-2 bg-[#fafafa] rounded text-sm">
          <span class="font-bold text-[#1a1a1a]">IDQN</span>
          <span class="text-[#666] ml-1">Independent baseline</span>
        </div>
        <div class="p-2 bg-[#fafafa] rounded text-sm">
          <span class="font-bold text-[#1a1a1a]">QMIX</span>
          <span class="text-[#666] ml-1">Monotonic mixing</span>
        </div>
        <div class="p-2 bg-[#fafafa] rounded text-sm">
          <span class="font-bold text-[#1a1a1a]">VDN</span>
          <span class="text-[#666] ml-1">Additive decomposition</span>
        </div>
        <div class="p-2 bg-[#fafafa] rounded text-sm">
          <span class="font-bold text-[#1a1a1a]">QPLEX</span>
          <span class="text-[#666] ml-1">Duplex dueling</span>
        </div>
      </div>
    </div>
    <div>
      <h3 class="text-lg font-semibold text-[#888] mb-3">Policy Gradient</h3>
      <div class="space-y-2">
        <div class="p-2 bg-[#fafafa] rounded text-sm">
          <span class="font-bold text-[#1a1a1a]">IPPO</span>
          <span class="text-[#666] ml-1">Independent baseline</span>
        </div>
        <div class="p-2 bg-[#fafafa] rounded text-sm">
          <span class="font-bold text-[#1a1a1a]">MAPPO</span>
          <span class="text-[#666] ml-1">Shared critic</span>
        </div>
        <div class="p-2 bg-[#fafafa] rounded text-sm">
          <span class="font-bold text-[#1a1a1a]">HAPPO</span>
          <span class="text-[#666] ml-1">Sequential updates</span>
        </div>
      </div>
    </div>
    <div>
      <h3 class="text-lg font-semibold text-[#888] mb-3">Training Features</h3>
      <div class="space-y-2">
        <div class="p-2 bg-[#fafafa] rounded text-sm">
          <span class="font-bold text-[#1a1a1a]">Automated Curriculum</span>
          <p class="text-xs text-[#666] mt-1">Progressive scenario difficulty</p>
        </div>
        <div class="p-2 bg-[#fafafa] rounded text-sm">
          <span class="font-bold text-[#1a1a1a]">Batch Training</span>
          <p class="text-xs text-[#666] mt-1">Train across multiple scenario variants</p>
        </div>
      </div>
    </div>
  </div>

  <p class="text-center text-sm text-[#888] mt-6">
    7 algorithms integrated with > 50,000 FPS on standard computer
  </p>
</div>

---

<div class="h-full flex flex-col justify-center items-center text-center px-16">
  <h1 class="text-6xl font-bold text-[#1a1a1a] mb-6">COMRAD</h1>

  <p class="text-xl text-[#666] mb-8 max-w-2xl">
    A standardized, procedurally-generated benchmark for studying cooperative emergent behaviors in 3D first-person environments.
  </p>

  <div class="grid grid-cols-4 gap-6 text-sm text-[#888] mb-10">
    <div>13+ scenarios</div>
    <div>4 behavior categories</div>
    <div>7 baseline algorithms</div>
    <div>>50K FPS</div>
  </div>

  <p class="text-lg text-[#999]">Any Questions?</p>
</div>
