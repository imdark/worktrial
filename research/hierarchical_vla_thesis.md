# Planner and Controller for Two-Finger Manipulation: Models, Costs, and How to Connect Them

**Status:** Research findings v1 · 2026-09-24
**Scope:** Which vision-language models (VLMs) and vision-language-action models (VLAs) to use for a hierarchical manipulation policy on a single arm with a two-finger parallel gripper (SO-101 placeholder, LeRobot, MuJoCo, Apple M5 / 24 GB first, cloud GPU later). It covers cheap vs. expensive choices, and how to pair a slow, capable planner with a fast feedback controller.
**Companion to:** [`../DESIGN.md`](../DESIGN.md), Stages 5–10.

---

## How this was researched, and how to read it

Four research agents worked in parallel on four questions:
1. the high-level planner,
2. the low-level controller,
3. how to connect the two levels,
4. cost and hardware.

Each agent cited primary sources: papers, model cards, pricing pages and READMEs. Their findings were then checked against each other; §10 lists where they disagreed and how that was resolved.

Four claims the recommendations depend on were re-checked by hand on 2026-09-24:

| Claim | Checked against | Result |
|---|---|---|
| SO-101 benchmark numbers | arXiv 2606.08881, Tables 3 and 5 | ✅ matches |
| Gemini Robotics ER 2 pricing | Google pricing page | ✅ matches |
| ACT has an `observation.environment_state` input slot | LeRobot source code | ✅ confirmed |
| Development machine | local system check | ✅ Apple M5 (10-core GPU), 24 GB |

Every number carries a tag:

| Tag | Meaning |
|---|---|
| **[V]** | Verified in a primary source (spec, config, license, pricing, or a third-party measurement with its method stated) |
| **[VC]** | Vendor claim: a performance number reported by the model's own authors |
| **[3P]** | Independent third-party measurement |
| **[E]** | Estimate. Arithmetic is shown or follows from stated assumptions. |
| **[U]** | Uncertain: a single weak source or unconfirmed |

> The field changes roughly monthly. More than half the models named here were released after mid-2025. Prices marked "promo" change on 2027-01-01. Re-check anything load-bearing before committing to it.

---

## 0. Summary of findings

1. **Split the "planner" into two jobs.**
   - A *semantic planner* decides what to do next. Run it when something happens (task start, subtask finished, failure), not on a timer. This is where expensive frontier models are worth paying for.
   - A *grounder* turns "pick the red cube" + an image into a pixel point, then a 3D point. It runs at ~0.5–2 Hz, and small open models are now as good at this as frontier ones. Molmo2-ER-4B scores 77.3 on Point-Bench against GPT-5's 43.6 [V].
   - Paying frontier prices for pointing wastes money.
2. **Calling the planner 5 times a second does not make it react in 200 ms.** Each planner call takes 1–2 s even without reasoning mode [V/E]. At 5 Hz you would have five requests in flight, each acting on a frame 1–2 s old, at five times the cost.
   - Reactivity has to come from the controller.
   - The target point should be *tracked* at controller rate between plans, not re-queried.
3. **The fast loop must run on the machine next to the robot.** The budget per step is 10–33 ms. The fastest cloud APIs take 0.45–1.1 s just to start answering [V].
4. **Keep your typed interface, `{subtask, target_point_3d, phase}`.** 2026 research supports it:
   - Grounded points or boxes beat language-only commands by a large margin: 92.5% vs 32.4% in Point-VLA [V].
   - Google DeepMind's orchestration study found that deciding *when a subtask is finished* and *how well the controller follows commands* matter more than typed vs. latent [V].
   - Latent interfaces (Helix, GR00T) need joint training data on the order of hundreds of hours.
5. **On an SO-101, a large pretrained VLA clearly beats ACT, and SmolVLA does not.** With 100 demos per task [V, arXiv 2606.08881]:

   | Policy | Success | Recovery rate |
   |---|---|---|
   | π0.5 | 56.25% | 30.8% |
   | Wall-X | 51.25% | — |
   | ACT | 33.75% | 6.5% |
   | SmolVLA | 32.5% | 3.2% |

   SmolVLA's advertised advantage over ACT did not reproduce. It is still useful as the only language-conditioned VLA that runs on a Mac. It is not an accuracy upgrade.
6. **Your two-finger gripper labels your demos for free.**
   - Its open and close events split each episode into approach → grasp → transport → place → retreat.
   - The end-effector position at the next open/close event is a free hindsight label for `target_point_3d`.
   - Recent work using this method: 90.5% of automatically found segment boundaries were accepted by human reviewers [V, arXiv 2609.24059].
7. **Stage 10 (V-JEPA 2-AC as the low level) will not work as written.** V-JEPA 2-AC plans at ~16 s per action and only accepts goal images [V]. Redefine it as a mid-level planner or checker (§7).
8. **Cost follows architecture, not model choice.** An event-driven planner at ~0.3 Hz on Gemini Robotics ER 2 costs about $1.7–3.5 per robot-hour. The same model polled at 5 Hz costs $28–58 per robot-hour [E]. §5 has the full table.

---

## 1. The problem: two loops with very different time budgets

```
              ┌─────────────────────────────────────────────┐
  events ───▶ │  SEMANTIC PLANNER  (frontier VLM, 0.05–0.3 Hz)│  "what next?"
              └───────────────┬─────────────────────────────┘
                              │ subtask (str), phase
              ┌───────────────▼─────────────────────────────┐
              │  GROUNDER  (small open VLM, 0.5–2 Hz)        │  "where exactly?"
              │  2D point ─▶ depth lift ─▶ target_point_3d   │
              └───────────────┬─────────────────────────────┘
                              │ PlannerCommand (latest-value slot)
              ┌───────────────▼─────────────────────────────┐
              │  TRACKER  (object pose / point tracker, ≥10 Hz)│  keeps target fresh
              └───────────────┬─────────────────────────────┘
                              │
              ┌───────────────▼─────────────────────────────┐
              │  CONTROLLER  (ACT / SmolVLA / π0.5, 30–50 Hz)│  action chunks + RTC
              └───────────────┬─────────────────────────────┘
                              │ joint targets, gripper [0,1]
              ┌───────────────▼─────────────────────────────┐
              │  control/safety.py  (deterministic clip, 30+ Hz)│
              └─────────────────────────────────────────────┘
```

| Layer | Rate | Latency budget | Must run | Why |
|---|---|---|---|---|
| Safety / clipping | every step | &lt;1 ms | Local, deterministic | Can't be learned or remote |
| Controller | 30–50 Hz execution; one inference per chunk | chunk_len / fps (e.g. 50/30 = 1.67 s per chunk with async inference) | Local, or a GPU on the same LAN | WAN round trips of ~300 ms caused jerky control at 5–12 Hz vs 20 Hz locally [V] |
| Tracker | ≥10 Hz | ~50–100 ms | Local | Stops the target point going stale between grounder calls |
| Grounder | 0.5–2 Hz | 0.5–2 s | Mac or cloud | Small model; latency is tolerable |
| Semantic planner | event-driven | 1–10 s | Cloud API is fine | Called rarely; reasoning quality matters more than speed |

---

## 2. High-level planner

### 2.1 Options by cost tier

**Top capability, high cost**

| Model | Open? | Strengths | Embodied evidence | Price in / out ($/M tokens) |
|---|---|---|---|---|
| **Gemini Robotics ER 2** (Jul 2026, preview) | Closed | Only frontier model trained as a robot orchestrator. Outputs points, boxes and trajectories; detects success and progress; calls VLAs as tools; has a streaming endpoint. | ERQA 78.5; success detection 87.7 (image) / 82.4 (video); progress 57.4 [VC] | **1 / 5** until 2026-12-31, then 2 / 10 [V] |
| Claude Opus 5.5 | Closed | Long-horizon decomposition, recovering from failures, conversation | Opus 5 ERQA 67.2, as measured by Google [VC] | 4 / 20 [V] |
| GPT-6 Sol | Closed | Same | GPT-5 ERQA 59.0; Point-Bench only 43.6 [V] | 2 / 10 [V] |
| Gemini 3.1 Pro | Closed | Same | Gemini-3-Pro RefSpatial 65.5 [V] | 2 / 12 [V] |

**Mid**

| Model | Open? | Strengths | Embodied evidence | Price in / out ($/M tokens) |
|---|---|---|---|---|
| Gemini 3.8 Flash / 3.5 Flash-Lite | Closed | Cheap general planning | ER 1.6 (a Flash base) beat Flash on instrument reading [VC] | 0.75 / 3.75 (promo); 0.30 / 2.50 [V] |
| Claude Sonnet 5, GPT-6 Luna | Closed | Cheap semantic planning | No published pointing benchmarks | 2 / 10; 0.10 / 0.50 [V] |
| Qwen3.6-27B / 3.8-27B | Apache-2.0 | Best open ERQA | ERQA 62.5 / 65.5, RefSpatial 70.0 [V] | Self-host (~$0.42 / 3.00 on OpenRouter) |
| Cosmos Reason 2-8B | NVIDIA Open Model License (commercial use allowed) | 2D/3D point localization | Robotics score 56.9 vs 53.1 for its base model [VC] | Self-host, needs ≥32 GB of GPU memory |

**Low cost, local, open**

| Model | Open? | Strengths | Embodied evidence | Runs on Mac? |
|---|---|---|---|---|
| **Molmo2-ER-4B** (Ai2) | Apache-2.0; weights, data and code | Best open pointing; backbone of MolmoAct2 | Point-Bench 77.3, Where2Place 54.0, 13-benchmark average 63.8 vs ER 1.5's 55.0 [V] | Architecture supported in `mlx-vlm`; no ER-specific conversion confirmed [U] |
| **RoboBrain 2.5-4B / 8B** (BAAI) | Apache-2.0 | The only open model that directly outputs points with depth (u, v, depth) and 3D traces | 8B: RoboSpatial 73.0, TraceSpatial 44% vs Gemini-3-Pro's 7% [V] | Probably, via the Qwen3-VL path [U] |
| Qwen3-VL-8B | Apache-2.0 | Most robust general open VLM, with first-class tooling | Where2Place 63.0, Point-Bench 64.2 [V] | **Yes** (mlx-vlm, llama.cpp, Ollama) [V] |

**Legacy, do not build on these:**
- Gemini Robotics-ER 1.5 and 1.6: being deprecated, and absent from the current pricing page [V].
- RoboPoint, SpatialVLM, Magma, PaliGemma 2.

**Benchmark caveat:** scores come from different evaluation harnesses. One audit found the same vendor report giving Gemini 2.5 Pro two different Where2Place scores (37.0 and 22.0) [U]. Compare models within one paper's table, not across papers.

### 2.2 Recommendation

- **Semantic planner:**
  - Default: Gemini Robotics ER 2, called on events.
  - Use Claude Opus 5.5 or GPT-6 Sol at &lt;0.1 Hz only when tasks need long-horizon reasoning or recovery from unusual failures.
  - Turn reasoning off or to its lowest setting. Reasoning tokens are billed as output: 300 of them on Opus 5.5 add ~$21.6/h at 1 Hz [E]. They also push latency from ~1 s to 8–100 s [V].
- **Grounder:**
  - Molmo2-ER-4B running locally, with RoboBrain 2.5 as a cross-check for depth-aware outputs.
  - Use ER 2 as the paid baseline to measure the local models against.
- **Success detection:** don't rely on a VLM alone.
  - FailBench (Sep 2026): the best of 13 VLM failure detectors reached only 0.77 balanced accuracy.
  - They lean toward calling ambiguous cases a success, and drop below 0.60 on contact-heavy tasks [V].
  - Combine the VLM verdict with proprioceptive checks (§4.4).

### 2.3 From a 2D point to `target_point_3d` for a two-finger gripper

1. **Get pixel coordinates.** Gemini and Qwen return `[y, x]` scaled to 0–1000, so u = x/1000·W and v = y/1000·H. Molmo uses its own point format.
2. **Get depth.** Take the median of a 5×5 patch of the aligned depth image, rejecting zero and NaN values.
   - MuJoCo: render depth natively, with fy = H / (2·tan(fovy/2)). MuJoCo cameras look down −z, so flip axes to match the OpenCV convention.
   - Real arm: RealSense D405/D435 or OAK-D.
   - No depth camera: Depth Anything 3 Metric-Large (Apache-2.0) or MoGe-3 (~121 ms per frame) [V/VC].
3. **Back-project.** X = (u−cx)·z/fx, Y = (v−cy)·z/fy. Then apply T_base←cam from a ChArUco calibration (DESIGN.md Stage 7).
4. **Correct for where the gripper needs to be.** A VLM point sits on the visible surface, but a parallel gripper needs the grasp center.
   - Prompt SAM 3 with the point to get an object mask.
   - Take the masked point cloud and use its centroid or top-surface point.
   - PCA of that cloud gives `approach_yaw` and `gripper_width_hint`.
5. **Native-3D models** (RoboBrain 2.5 u,v,depth; Cosmos Reason 2 3D points; Qwen3-VL 9-DoF boxes) skip the lifting step, but they guess depth. Use them to cross-check, not as ground truth.

---

## 3. Low-level controller

### 3.1 Options by cost tier (Stage 5+ candidates)

**Low cost, fast (runs on a Mac)**

| Policy | Params | Action head | Latency | Fine-tune cost | Demos | Can accept planner commands via |
|---|---|---|---|---|---|---|
| **ACT** | ~52–80M | CVAE + L1 regression, chunked | 5.0 ms on 4090; **42.7 ms MPS on M1**; 182 ms on M1 CPU [V] | 100k steps ≈ 3.3 h on a 5070 Ti ≈ **~$1 rented** [E]; overnight on a Mac [U] | ~50 [V] | **`observation.environment_state`**, a dedicated encoder token [V, code]. No language input. |
| Diffusion Policy | ~250M [U] | DDIM | ~100 ms on a 3080 with 10 steps [VC] | ~$2 per 100k steps [E] | 100–200 [U] | Global conditioning vector |
| VQ-BeT | small | Residual VQ, one forward pass | 5–25× faster than Diffusion Policy [VC] | Low | 50–100 [U] | Goal-conditioned variant |

**Mid**

| Policy | Params | Action head | Latency | Fine-tune cost | Demos | Can accept planner commands via |
|---|---|---|---|---|---|---|
| **SmolVLA** (Apache-2.0) | 450M | Flow matching, 50-action chunk | 99 ms on 4090; **205 ms MPS / 111 ms MLX on M5 Pro** [3P]; 722 ms MPS on M1 [V] | 20k steps ≈ 4 h on an A100 ≈ **$5–10** [V/E] | ~50; 25 was too few [V] | Language + extra state dimensions [U] |
| X-VLA-0.9B (Apache-2.0) | 0.9B | Flow matching, chunk 32 | Not published | Soft-prompt tuning of 9M params ≈ π0 [VC] | Few [VC] | Language + `domain_id` |
| EVO-1 / VLA-Adapter | 0.77B / 0.5B | Flow / regression | 16.4 Hz [VC] / 36 ms on H100 [VC] | Fits a 24 GB card | — | Language |

**High cost, high capability (CUDA only)**

| Policy | Params | Action head | Latency | Fine-tune cost | Demos | Can accept planner commands via |
|---|---|---|---|---|---|---|
| **π0.5** (openpi, Apache-2.0 [V]; weights licensing flagged [U]) | ~3.3B | Flow-matching expert, 50-action chunk | **~200 ms in LeRobot as shipped** [3P]; 73–76 ms in the paper [VC]; **17.6 ms on 5090 / 44 ms on Jetson Thor with FlashRT** [3P] | LoRA needs &gt;22.5 GB (4090); full fine-tune &gt;70 GB. 30k full-FT steps ≈ 12–17 h on H100 ≈ **$32–46** [E] | 100 per task in the SO-101 benchmark [V] | **Subtask language (trained on subtask commands)** + state tokens [V] |
| **GR00T N1.7** (commercial-OK weights) | 3B | Flow matching, 4 denoising steps | **27.9 ms with TensorRT** on H100 / RTX Pro 6000 [VC]; 216 ms on Jetson Orin [VC] | ≥40 GB; ~$2–10 per SO-101 run [U/E] | 80 episodes (N1.5 SO-101 demo) [V] | Language + embodiment tag + state |
| MolmoAct2 (Apache-2.0, fully open) | ~4B | Flow matching + depth/trace reasoning | p50 227 ms on 5090 [VC] | LoRA on one 24 GB GPU [V] | Has an SO-100/101 checkpoint | Language; the planned path it outputs can be edited |

**Avoid for this project:**
- **GO-1:** CC BY-NC-SA, non-commercial [V].
- **RDT2:** zero-shot needs the UMI gripper [V].
- **OpenVLA-OFT:** fine-tuning takes 8× A100 [V].
- **π0-FAST as the fast loop:** it decodes actions token by token, ~750 ms per chunk [U].

**Closed, for reference:**
- **π0.7:** subtask + subgoal-image conditioning, trained with simulated 0–240 ms delays.
- **Gemini Robotics On-Device 2:** trusted testers only. 53.3% on SO-101 tasks [VC].

### 3.2 Latency on your hardware (Apple M5, 10-core GPU, 24 GB)

No measurements exist for a base M5. These are estimated [E] by scaling the M1 and M5 Pro measurements by GPU core count and memory bandwidth.

| Policy | Estimated M5 latency | At 30 Hz execution |
|---|---|---|
| ACT (MPS) | ~20–30 ms per inference | Works, but **don't use temporal ensembling**: it runs inference every step, leaving no headroom. Execute chunks with `n_action_steps` > 1, or use async inference. |
| SmolVLA (MPS / MLX) | ~300–450 ms / ~180–250 ms per 50-action chunk | Works with async inference or RTC; a 50-action chunk lasts 1.67 s. |
| π0.5 / GR00T | Not practical (&gt;5 s or CUDA-only) | Needs an NVIDIA GPU or Jetson Thor. |

⚠️ **Shared-GPU contention.** If the grounder VLM runs on the same M5 as the controller, they compete for its GPU and memory bandwidth.
- On a base M5, pair a local grounder with ACT, not SmolVLA.
- Or move the grounder to the cloud.

### 3.3 Techniques that keep the fast loop running

- **Async inference (LeRobot PolicyServer / RobotClient).** Works with every LeRobot policy.
  - Settings: `actions_per_chunk` 10–50, `chunk_size_threshold` 0.5–0.6.
  - SmolVLA paper result: task time down 30%, and 19 tasks completed vs 9 in a fixed time window [VC].
- **Real-Time Chunking (RTC).** Blends the start of each new chunk into the actions already committed, so there are no jumps at chunk boundaries.
  - In LeRobot for π0, π0.5, SmolVLA, EVO-1, and GR00T N1.7 rollouts [V].
  - Settings: `execution_horizon` 8–12. Costs ~21 ms on a 4090 for π0.5 [VC].
- **Training-time RTC** (`policy.rtc_training_max_delay` for π0.5) trains the model on simulated delay instead, so it adds nothing at inference [V]. Prefer it when available.
- **Temporal ensembling** failed at higher inference delays and triggered protective stops in the RTC paper's tests [VC]. Use it only with ACT when inference is well under one step.
- **Implementation matters more than model choice.** π0 on the same 4090 ranges from ~200 ms (LeRobot as shipped) to 20 ms (CUDA graphs + kernel fusion) [3P]. Measure your actual stack.

### 3.4 Recommendation

| Phase | Controller | Planner-command conditioning | Why |
|---|---|---|---|
| Mac, sim (Stage 5) | **ACT** | `environment_state = [target_xyz, phase one-hot, subtask one-hot]` | Cheapest way to test whether the controller follows commands, before any VLA is involved |
| Mac, language (Stage 9a) | **SmolVLA** | Subtask as the language prompt, target appended to state | Only VLA that runs on a Mac; not more accurate than ACT |
| Cloud GPU (Stage 8–9) | **π0.5** via LeRobot, training-time RTC | Subtask language + state | Best measured SO-101 success and recovery; trained on subtask commands |
| Edge deployment / commercial | **GR00T N1.7** + TensorRT | Language + state | ~28 ms optimized; commercial-OK weights; Jetson Thor path |

---

## 4. Connecting the two levels

### 4.1 How leading systems do it

| System | Planner → controller | Interface | Planner Hz | Controller Hz | Open? |
|---|---|---|---|---|---|
| Figure Helix / Helix 02 | 7B VLM → 80M transformer (→ 10M full-body controller at 1 kHz) | Learned latent | 7–9 | 200 (1000) | No |
| GR00T N1.x | VLM → flow-matching action model | Middle-layer VLM tokens | 10 | 120 | Weights |
| Hi Robot (PI) | VLM → π0 | Atomic language commands | ~1 + on user input | ~50 [U] | No |
| π0.5 / π0.7 | Same model predicts the subtask, then actions | Subtask text (+ subgoal images in π0.7) | Low; subgoal every 4 s | Chunked | π0.5 yes |
| Gemini Robotics 1.5 + ER | ER orchestrates the VLA via tool calls | Language + points | Success detection at 5 Hz | — | ER via API |
| HAMSTER / 3D HAMSTER | Fine-tuned VLM → 3D policy | Path drawn on the image → 3D waypoints | Once or occasionally | Policy rate | Code |
| MolmoAct2 | One model: depth → path in the image → actions | Path the user can edit | Per chunk | 10–30 | Fully |
| Steerable Policies (Feb 2026) | Reasoner / Gemini 3 → steerable VLA | **Mixed:** subtask, motion, points, gripper paths | Per subtask | — | Research |
| Point-VLA | Multimodal model or human → VLA | Bounding box | Once | — | — |
| ReKep / VoxPoser | GPT-4o-class model → optimizer | Keypoints / code / cost functions | 5–10 re-optimization | Planner rate | Code |
| V-JEPA 2-AC | Goal images → search in latent space | Goal image | **~0.06 (16 s per action)** | Same | Weights |

Measured gains for richer interfaces:
- Hi Robot: +40% instruction accuracy vs a GPT-4o planner [V].
- Gemini ER orchestrator: progress score 80% vs 44% for the VLA alone [V].
- HAMSTER: +20% absolute over OpenVLA [V].
- Point-VLA: 92.5% vs 32.4% [V].

### 4.2 Interface types compared

| Interface | Best at | Weakness | Debuggability | Fit for two-finger gripper |
|---|---|---|---|---|
| Language subtask | Semantics; human corrections | *Where*: duplicate objects, precise placement | High | Needed, but not enough alone |
| **Points / keypoints** | Precise grounding; free hindsight labels | Needs depth and calibration; no orientation | High | **Core field** |
| 2D/3D path | Approach direction | Planner must be fine-tuned; 2D paths distort depth | High (overlay) | Stage 2 upgrade |
| Goal image | Very rich specification | Slow to generate; hard to edit | Medium | V-JEPA only |
| Learned latent | Maximum information; no labeling | Opaque; needs large joint data; can't swap planners | Low | Last, if ever |
| Code / constraints | No training; checkable | Brittle to perception errors | Very high | Scripted expert (Stage 4) |

### 4.3 Recommended interface (typed, versioned)

```python
@dataclass
class PlannerCommand:
    subtask: str                    # "pick up the red cube"
    subtask_id: int                 # index into a small, fixed vocabulary (<~15)
    phase: Phase                    # APPROACH | GRASP | TRANSPORT | PLACE | RETREAT
    target_point_3d: np.ndarray     # (3,) metres, robot base frame
    target_px: tuple[int, int]      # (u, v) in the source camera — debugging + re-grounding
    approach_yaw: float | None      # rad, from mask PCA
    gripper_width_hint: float | None  # normalised [0,1]
    confidence: float               # planner/grounder confidence
    obs_timestamp: float            # monotonic time of the frame it was computed from
    seq_id: int                     # increments on every new command
```

Three design rules:
1. **Latest value wins.** The planner writes, and the controller reads every tick and never blocks. Helix, τ0-VLA, Libra-VLA, DuoCore-FS and π0.7 all use this pattern [V].
2. **Train the controller to expect stale commands.** Pair a command computed at time *t* with observations from *t+Δ*, where Δ ~ U(0, measured grounder latency) (Helix, DuoCore-FS) [V].
3. **Drop fields at random during training** (~15–30% each, as π0.7 does). The controller then still works when the grounder has no point or the planner has no subtask [V].

Also draw `target_px` onto the controller's camera image as an overlay. Pretrained VLAs read image overlays more reliably than raw coordinates, which is the lesson from HAMSTER, RT-Trajectory and Point-VLA [U, inferred].

### 4.4 Replanning and success detection

When to call the planner again, in order of preference:
1. **Success detected.** The GDM orchestration study found this beats fixed timers and letting the VLM predict how long a step will take. It stayed robust with a 10–30% detector error rate, and 4–8 s execution windows worked best [V, arXiv 2606.10267].
2. **Gripper open/close transition.**
3. **Safety timer** (4–8 s).
4. **Target lost:** the tracker's point jumps by more than a threshold, or tracking confidence drops.
5. **User input.**

Success detection combines three signals:
- A VLM verdict on a cropped image. Cropping to the relevant region improves accuracy [V].
- **Proprioception:** the gripper stopped short of fully closed (an object is between the fingers); servo load (the SO-101's STS3215 servos report it); in sim, object height.
- **Text context for the planner:** give it text descriptions with bounding boxes plus contact information, not just raw images. This helps substantially [V].

### 4.5 Labeling subtasks in demos

Record these at Stage 3, at no extra cost, because every label below can be computed from them:
- gripper command and measured width
- joint velocities
- per-sample timestamps (already in DESIGN.md)

| Label | How to compute it | Source |
|---|---|---|
| Segment boundaries | Gripper command changes sign + joint velocity near zero | PerAct / C2F-ARM keyframe method [V] |
| `phase` | Position relative to gripper events (before close = APPROACH/GRASP, etc.) | Derived |
| `target_point_3d` | End-effector position at the *next* gripper event | Hindsight labeling [V: RT-Trajectory, Point-VLA] |
| `subtask` text | Template in sim; a VLM names each gripper-anchored segment on real data (90.5% of boundaries accepted by reviewers) | [V, arXiv 2609.24059] |
| Motion labels (optional) | End-effector deltas + gripper → "move forward / close gripper" | RT-H [V] |
| Robustness | Randomly shift boundaries ±N frames in training | SparkVLA [V] |

**Warning from the GDM study:** fine-tuning the controller on sim data *reduced how well it followed commands* [V, 2606.10267]. Check command-following explicitly before moving to the real arm (§7, Stage 8).

---

## 5. Costs

### 5.1 Planner API cost per robot-hour

Call assumptions: 500 text tokens + 2 × 640×480 images in, 100 tokens out, no reasoning, no caching.

Image tokens per frame [V]:
- Claude: 414.
- GPT-5.x: 360.
- Gemini 3.x: 1,120 at the default high resolution, 280 with `media_resolution: low`.
- Qwen3-VL: 300.

$/h = per-call cost × calls per hour. All results below are [E] from verified prices.

| Model | 1 Hz polling, 2 images | **Event-driven, ~0.3 Hz avg** | 5 Hz polling |
|---|---|---|---|
| Local Molmo2-ER / Qwen3-VL on the Mac | $0 | $0 | Not achievable |
| Qwen3-VL-8B (OpenRouter) | $0.63 | $0.19 | $3.14 |
| GPT-6 Luna | $0.62 | $0.19 | $3.10 |
| Gemini 3.5 Flash-Lite | $3.86 | $1.16 | $19.30 |
| Gemini 3.8 Flash, low res | $4.21 | $1.26 | $21.06 |
| **Gemini Robotics ER 2, low res** | **$5.62** | **$1.69** | $28.08 |
| Gemini Robotics ER 2, high res | $11.66 | $3.50 | $58.32 |
| Claude Haiku 4.5 | $6.58 | $1.97 | $32.90 |
| GPT-6 Sol | $12.38 | $3.71 | $61.92 |
| Claude Sonnet 5 | $13.16 | $3.95 | $65.81 |
| Claude Opus 5.5 | $26.32 | $7.90 | $131.62 |

Notes on the table:
- **ER 2 price change:** ER 2 prices double on 2027-01-01 [V].
- **Batch APIs:** half price. Useless in a live loop, but ideal for the offline labeling in §4.5.
- **Caching:** does nothing here, because a 500-token prompt is below every provider's minimum cacheable size [V].

**Latency without reasoning, full call with 2 images** [V time-to-first-token + E]:

| Model | Latency |
|---|---|
| Gemini Flash-class | ~1.0 s |
| Claude Haiku 4.5 | ~1.8 s |
| GPT-5.6 Sol | ~2.6 s |
| Sonnet 5, high reasoning | Time to first token alone is 8.5 s |
| GPT-6 Luna, max reasoning | ~105 s |

### 5.2 Running the grounder on your M5 (24 GB)

This is estimated [E] by scaling measured Apple Silicon throughput to the base M5's ~150 GB/s memory bandwidth. A comparable measured data point: Qwen3.5-9B at 4-bit on an M4 Pro took ~3.5 s to first token with one large image, then ran at ~50 tokens/s [V].

| Model (4-bit) | Size in memory | Per call (1 image, ~30-token JSON out) | Achievable rate |
|---|---|---|---|
| Molmo2-ER-4B | ~3 GB | ~1–2.5 s | ~0.4–1 Hz |
| Qwen3-VL-8B | ~5.5 GB | ~2–4 s | ~0.25–0.5 Hz |

Enough for an event-driven grounder, not for 5 Hz. Two ways to go faster:
- Cache the text prompt so only the image tokens are processed on each call.
- Emit short JSON.

### 5.3 Controller hardware and rental

| Option | ACT | SmolVLA | π0.5 | GR00T N1.7 | Cost |
|---|---|---|---|---|---|
| Your M5 (MPS) | ~20–30 ms [E] | ~300–450 ms per chunk [E] | ✗ | ✗ | Owned |
| RTX 4090 (rented) | 5 ms | 99 ms | 70–200 ms (18–20 ms optimized) | — | **$0.34/h** community, $0.74 secure [V] |
| RTX 5090 (rented) | — | — | 17.6 ms FlashRT | 13 ms (N1.6) | $0.69–0.99/h [V]; buying: $4.3–5.2k street [V] |
| H100 (rented) | — | — | — | 27.9 ms TensorRT | $1.99–3.49/h [V] |
| Jetson AGX Thor | — | — | 44 ms FlashRT | 31 ms (N1.6) | **$5,499** (raised from $3,499 in Jul 2026) [V] |

Fine-tuning cost per run [E, from verified step times and $/h]:

| Policy | Where | Cost per run |
|---|---|---|
| ACT | Rented 4090 | ~$1–5 |
| SmolVLA | Rented A100 / 4090 | ~$2–10 |
| π0.5, LoRA | 4090 | ~$1–6 |
| π0.5, full fine-tune | H100 | ~$32–46 |
| GR00T N1.7 | L40S / A100 | ~$2–10 |

### 5.4 Three budgets

Assumes 40 robot-hours per month. The arm (SO-101 leader + follower + 2 USB cameras) is **~$200–450** one-time [V/E].

| | **A. Mac only** | **B. Mid** | **C. High** |
|---|---|---|---|
| Semantic planner | Local Qwen3-VL-8B, or GPT-6 Luna / Qwen via API | Gemini Robotics ER 2 (low res), event-driven | Opus 5.5 / GPT-6 Sol at 0.2 Hz + Flash-Lite monitoring at 1 Hz |
| Grounder | Molmo2-ER-4B on the M5 | Molmo2-ER-4B on the M5, or ER 2 | ER 2 or a self-hosted RoboBrain 2.5 |
| Controller | ACT (+ SmolVLA experiments) on the M5 | ACT / SmolVLA local; π0.5 LoRA trained on rented 4090s | π0.5 / GR00T local on a 5090 or Jetson Thor at 22–80 Hz |
| Training | Mac overnight + occasional $2–10 rental | ~60 GPU-h/month ($20–41) + 1–2 π0.5 full fine-tunes ($32–46 each) | Burst H100 / B200 |
| **Monthly** | **~$0–30** | **~$100–200** | **~$600–1,000** (hybrid planner ~$370 + hardware amortized ~$230–290 + bursts) |
| Buys you | Complete hierarchy end to end; no large VLA | Robotics-tuned planner + π0.5, the best measured SO-101 controller | Frontier planner + large VLA running locally at high rate |

**Recommendation:** start at A and move to B at Stage 8. B is where the measured accuracy gain is (π0.5 vs ACT: 56% vs 34%). C mostly buys planner intelligence, and a two-finger pick-and-place task rarely needs it.

---

## 6. Recommended architecture

**Default stack (Budget B):**

```
Gemini Robotics ER 2   ── event-driven ─────▶ subtask, phase, success verdict
Molmo2-ER-4B (M5)      ── 0.5–1 Hz ─────────▶ 2D point ─▶ depth + SAM mask ─▶ target_point_3d, yaw, width
Tracker (sim pose / SAM2) ── ≥10 Hz ────────▶ keeps target_point_3d fresh
Controller             ── 30 Hz execution ──▶ ACT (Mac) → π0.5 + RTC (GPU)
control/safety.py      ── every step ───────▶ deterministic clip + e-stop
```

**How to know it's working:** each level can be tested alone, as DESIGN.md Stage 9 requires.
- Drive the controller with *ground-truth* `PlannerCommand`s from MuJoCo and measure how well it follows them.
- Drive the planner/grounder against *recorded* frames and measure point error in centimetres.
- Only then connect them.

---

## 7. Changes proposed to DESIGN.md

| Stage | Current plan | Proposed change | Reason |
|---|---|---|---|
| 3 · Recording | Per-sample timestamps, commanded actions | Also record **gripper transition events** and measured gripper width as first-class fields | Needed for free labeling of `phase` / `target_point_3d` (§4.5) |
| 4 · Scripted expert | IK reach → grasp → lift → place | Have it emit `PlannerCommand` ground truth alongside actions | Sim-oracle labels for the Stage 5 conditioning experiment |
| 5 · Train/eval | ACT | ACT **conditioned on `environment_state` = target + phase + subtask**; add a *command-following* metric (send a point on a different object and check the arm goes there) | Tests the hierarchy's interface before any VLM exists |
| 5 · Train/eval | Temporal ensembling (implicit) | Use chunked execution or async inference on the Mac | Ensembling needs one inference per step; no headroom on M5 |
| 7 · Remote execution | ZMQ client/server | Evaluate **LeRobot async inference (gRPC PolicyServer)** before writing your own; log round-trip time | Solves the same problem; supports RTC |
| 8 · Sim2real | Randomization + co-training | Add an explicit **command-following regression check** before and after real fine-tuning | GDM found sim fine-tuning reduced command-following |
| 9 · Two-level policy | One VLM at 1–5 Hz | **Split into semantic planner (event-driven) + grounder (0.5–2 Hz) + tracker (≥10 Hz)**; latest-value command slot; stale-command training; random field dropout | §0.1–0.2; cost and latency |
| 9 · Success | (unspecified) | Hybrid VLM + proprioceptive success detector; success-triggered replanning | FailBench 0.77 ceiling; GDM study |
| 10 · World model | V-JEPA 2-AC **as the low level** | Recast as a **mid-level goal-image planner or subtask scorer** (0.1–1 Hz). Compare against the Stage 9 planner, not against ACT. | 16 s per action. Needs post-training on SO-101 data (trained on Franka, 7-D end-effector delta), and is camera-sensitive [V/U]. Meta's HWM (70% vs 0%, 3× less planning compute) and JEPA-WAM (87.3%) are the follow-ups to read. |

### Where Jev and Laya fit (from the earlier review)

Both are text-in, non-autoregressive "decision" models. Neither is a VLA.
- **Laya** (Apache-2.0, ~33 ms on a T4) is useful **offline**: batch-labeling episode failure modes and success, and deciding when to escalate to the expensive planner. Constraints:
  - It must be fine-tuned first: zero-shot it scores 0.36, barely above the 0.32 random baseline.
  - Keep choice sets under ~15 options.
- **Jev** is API-only, so skip it for anything on the robot.
- **laya-vision** (the image fork) has non-commercial weights and scores ~0 on CartPole and maze tasks. Do not use it for control.

---

## 8. Risks and open questions

| Risk | Likelihood | Mitigation |
|---|---|---|
| Grounder on the same M5 slows the controller | High on a base M5 | ACT when the grounder is local; otherwise move the grounder to the cloud; measure both |
| Gemini ER model churn (1.5 → 1.6 → 2 in 10 months, older versions deprecated) | High | Keep the planner behind the `Policy` seam; keep a local open-model fallback always working |
| Price changes on 2027-01-01 (ER 2 and 3.8 Flash double) | Certain | Budget at 2027 prices; event-driven calling keeps it small |
| Controller ignores the planner (poor command-following) | Medium | Command-following metric from Stage 5; field dropout; overlay target on the image |
| π0.5 latency in LeRobot as shipped (~200 ms) differs from the paper (73 ms) | Measured | Plan on 200 ms + RTC; optimize kernels only if needed |
| VLM success detection is overconfident | Measured (FailBench) | Hybrid detector; crop images; proprioceptive checks |
| Execution failures (unstable grasps, repetition loops) dominate on SO-101 for **every** model | Measured [V] | Budget failure and recovery demos; DAgger (`lerobot-rollout --strategy.type=dagger`) |
| Licensing | — | Commercial-OK: π0.5 / openpi (Apache-2.0; weights flagged [U]), GR00T (NVIDIA OML), Molmo2 / RoboBrain / Qwen (Apache-2.0). **Non-commercial:** GO-1, laya-vision weights. Check before shipping. |

**Open questions for you:**
1. Will this be commercial? That rules out GO-1 and laya-vision, and decides whether GR00T's license matters.
2. Depth camera on the real rig (RealSense / OAK-D) or monocular with Depth Anything 3? This affects the accuracy of `target_point_3d`.
3. Is a local NVIDIA box likely (Budget C), or cloud-only (Budget B)?

---

## 9. Suggested next experiments (cheapest first)

1. **Stage 5 command-following test (Mac, $0).**
   - Scripted expert emits `PlannerCommand`; train ACT with it as `environment_state`.
   - Measure success *and* whether the arm goes where commanded when given a point on a different object.
2. **Grounder accuracy benchmark (Mac, $0; + ~$1 of ER 2).**
   - Render 200 MuJoCo frames with known object poses.
   - Compare Molmo2-ER-4B, Qwen3-VL-8B and ER 2 on 3D error after lifting (cm) and latency on the M5.
3. **Success-detector check ($0).** On 100 scripted episodes with known outcomes, compare VLM-only, proprioception-only and hybrid detectors.
4. **π0.5 LoRA on a rented 4090 (~$5–15)** once ~100 real SO-101 demos exist. Compare against ACT on the same evaluation seeds.

---

## 10. Where the sources disagreed

| Topic | Disagreement | Resolution |
|---|---|---|
| π0 / π0.5 latency on a 4090 | 73–76 ms (paper) vs ~200 ms (LeRobot issue #1537) vs 18–20 ms (hand-optimized) | All three are real; the implementation decides. Plan on the shipped 200 ms. |
| SmolVLA vs ACT | Vendor: SmolVLA > ACT. Third party on SO-101: 32.5% vs 33.75%, recovery 3.2% vs 6.5% | Trust the independent SO-101 benchmark (verified, Table 3/5); use SmolVLA for language input, not accuracy |
| MolmoAct2 "56.7% zero-shot on SO-100/101" vs π0.5 56.25% | Look equal | Different tasks and harnesses; **not comparable** |
| "5 Hz planner" | The design target implied reaction within 200 ms | Call latency is 1–2 s, so 5 Hz means stale pipelined calls. Replaced with event-driven planner + grounder + tracker. |
| Planner cost at 1 Hz with ER 2 | Planner agent ~$8/h; cost agent $7.63–11.66/h | Consistent. Difference = 1 vs 2 images at default high resolution. Low resolution cuts it ~50%. |
| Local planner latency | 1–3 s (planner agent) vs ~4 s on M4 Pro (cost agent) | For a base M5: ~1–2.5 s for a 4B grounder, ~2–4 s for an 8B model [E] |
| π0.5 license | Apache-2.0 [V, controller agent] vs flagged uncertain [architecture agent] | openpi code is Apache-2.0; confirm the weights' terms before commercial use |

---

## Sources

**Planner VLMs:**
- [Gemini Robotics ER 2 blog](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)
- [Gemini robotics docs](https://ai.google.dev/gemini-api/docs/robotics-overview)
- [ER 2 benchmarks](https://deepmind.google/models/gemini-robotics/embodied-reasoning/)
- [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Molmo2-ER-4B](https://huggingface.co/allenai/Molmo2-ER-4B)
- [MolmoAct2 / Molmo2-ER paper](https://arxiv.org/html/2605.02881v1)
- [RoboBrain 2.5](https://arxiv.org/html/2601.14352)
- [Qwen3-VL report](https://arxiv.org/abs/2511.21631)
- [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
- [RynnBrain 1.1](https://arxiv.org/html/2607.17977v1)
- [Embodied-R1.5](https://arxiv.org/html/2606.11324)
- [mlx-vlm models](https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models)
- [Depth Anything 3](https://github.com/bytedance-seed/depth-anything-3)
- [MoGe-3](https://arxiv.org/abs/2607.17967)
- [Benchmark audit](https://blog.pebblous.ai/report/physical-ai-benchmark-redundancy-audit-2026-08/en/)

**Controllers:**
- [SO-101 VLA benchmark (arXiv 2606.08881)](https://arxiv.org/html/2606.08881)
- [SmolVLA paper](https://arxiv.org/html/2506.01844v1)
- [LeRobot v0.5](https://huggingface.co/blog/lerobot-release-v050)
- [LeRobot v0.6](https://huggingface.co/blog/lerobot-release-v060)
- [LeRobot RTC](https://huggingface.co/docs/lerobot/rtc)
- [LeRobot async](https://huggingface.co/docs/lerobot/async)
- [LeRobot π0.5](https://huggingface.co/docs/lerobot/pi05)
- [LeRobot GR00T](https://huggingface.co/docs/lerobot/groot)
- [ACT source](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py)
- [π0 latency issue #1537](https://github.com/huggingface/lerobot/issues/1537)
- [openpi](https://github.com/Physical-Intelligence/openpi)
- [π0.7](https://arxiv.org/html/2604.15483)
- [RTC](https://arxiv.org/html/2506.07339)
- [Training-time RTC](https://arxiv.org/pdf/2512.05964)
- [GR00T N1.7](https://huggingface.co/nvidia/GR00T-N1.7-3B)
- [Isaac-GR00T](https://github.com/Nvidia/Isaac-GR00T)
- [FlashRT Thor/5090 benchmarks](https://forums.developer.nvidia.com/t/real-time-inference-on-thor-rtx-pi0-5-gr00t-n1-6-1-7-thor-23-hz-rtx-5090-50-80hz/368788)
- [MolmoAct2 SO-100/101](https://huggingface.co/allenai/MolmoAct2-SO100_101)
- [X-VLA](https://github.com/2toinf/X-VLA)
- [OpenVLA-OFT](https://arxiv.org/html/2502.19645v1)
- [Gemini Robotics On-Device 2](https://deepmind.google/models/model-cards/gemini-robotics-on-device-2/)
- [mlx-smolvla benchmark](https://pypi.org/project/mlx-smolvla/0.1.2/)
- [VLA-Perf](https://arxiv.org/abs/2602.18397)

**Connecting the levels:**
- [Helix](https://www.figure.ai/news/helix)
- [GR00T N1](https://arxiv.org/abs/2503.14734)
- [Hi Robot](https://arxiv.org/abs/2502.19417)
- [π0.5](https://arxiv.org/abs/2504.16054)
- [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342)
- [What Matters in Orchestrating Robot Policies](https://arxiv.org/abs/2606.10267)
- [RT-H](https://arxiv.org/abs/2403.01823)
- [RT-Trajectory](https://arxiv.org/abs/2311.01977)
- [HAMSTER](https://arxiv.org/abs/2502.05485)
- [3D HAMSTER](https://arxiv.org/abs/2606.31329)
- [Steerable Policies](https://arxiv.org/abs/2602.13193)
- [Point-VLA](https://arxiv.org/abs/2512.18933)
- [ReKep](https://rekep-robot.github.io/)
- [VoxPoser](https://arxiv.org/abs/2307.05973)
- [τ0-VLA](https://arxiv.org/abs/2608.16885)
- [SparkVLA](https://arxiv.org/abs/2608.16172)
- [DuoCore-FS](https://arxiv.org/abs/2512.20188)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [HWM](https://arxiv.org/abs/2604.03208)
- [JEPA-WAM](https://arxiv.org/abs/2609.20277)
- [FailBench](https://arxiv.org/abs/2609.03611)
- [Gripper-anchored auto-labeling](https://arxiv.org/abs/2609.24059)
- [PerAct keyframes](https://arxiv.org/abs/2209.05451)

**Costs:**
- [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Anthropic vision tokens](https://platform.claude.com/docs/en/build-with-claude/vision)
- [OpenAI pricing](https://developers.openai.com/api/docs/pricing)
- [OpenAI vision tokens](https://developers.openai.com/api/docs/guides/images-vision)
- [Gemini media resolution](https://ai.google.dev/gemini-api/docs/media-resolution)
- [OpenRouter models](https://openrouter.ai/api/v1/models)
- [Artificial Analysis comparisons](https://artificialanalysis.ai/models/comparisons/gpt-6-luna-vs-claude-sonnet-5-high)
- [llama.cpp Apple Silicon](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [mlx-vlm M4 Pro measurement](https://github.com/Iito/spindll/issues/75)
- [LeRobot paper (latency tables)](https://arxiv.org/html/2602.22818v1)
- [lerobot-doctor](https://github.com/Yanshi-Robotics/lerobot-doctor)
- [Jetson-PI](https://arxiv.org/html/2607.12659)
- [GR00T hardware guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md)
- [RunPod](https://www.runpod.io/pricing)
- [Lambda](https://lambda.ai/pricing)
- [Modal](https://modal.com/pricing)
- [Jetson price increase](https://www.cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101/)
- [RTX 5090 pricing](https://www.pcgamesn.com/nvidia/rtx-5090-pricing-september-2026)
- [WAN teleop prototype](https://github.com/rocPAI-Forge/so101-simstudio/issues/10)
- [SO-ARM101](https://partabot.com/products/so-arm101)

**Jev / Laya (earlier review):**
- [Laya on Hugging Face](https://huggingface.co/convaiinnovations/laya)
- [Laya GitHub](https://github.com/NandhaKishorM/laya)
- [laya-vision](https://github.com/r33drichards/laya-vision)
- [TechCrunch on Jev](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/)
- [eesel Laya review](https://www.eesel.ai/blog/laya-ai-review)
