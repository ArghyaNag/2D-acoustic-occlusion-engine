# Acoustic Engine Design Decisions & Technical Architecture

This document records the foundational design decisions, mathematical and theoretical rationales, architectural trade-offs, and verification conventions governing the 2D Acoustic Occlusion Engine (internally designated the *2D Positional Audio Sandbox*).

---

## 1. Project Overview

### 1.1 Purpose and Identity
The project is a real-time 2D positional-audio sandbox developed for an undergraduate Signals and Systems curriculum at Bangladesh University of Engineering and Technology (BUET). Although occasionally referred to in early scaffolding as the "3D Positional Audio Sandbox," the engine operates strictly on a discrete two-dimensional grid ($40 \times 24$ cells). Spatialization models acoustic propagation across an azimuthal 2D plane: stereo horizontal panning, distance-law amplitude attenuation, obstacle diffraction, through-wall transmission, and discrete wall reflections, rather than 3D elevation or spherical harmonics.

### 1.2 The Four Operational Modes
The system was originally designed around four operational modes:
1. **Mode 1: Offline File-Based Sandbox (Built and Verified)**  
   Users load arbitrary audio files (`.wav`, `.flac`, `.ogg`, `.mp3`) onto source nodes via a native file dialog. Sources and listener can be freely positioned on the grid, wall layouts drawn or modified with multi-material swatches, and the resulting acoustic transformations observed in real time alongside live waveform and FFT spectrum visualizers.
2. **Mode 2: Local Microphone Scaffolding (Built and Verified)**  
   Integrates a live audio capture stream (`LiveInputStream` in `live_input.py`) via `sounddevice.InputStream`. Allows real-time microphone input to feed an active grid source.
3. **Mode 3: Multi-Device UDP Microphone Relay over LAN (Built and Verified)**  
   Implements a dedicated UDP packet receiver (`NetworkInputServer` in `network_input.py`) listening on port 50007. Remote clients run lightweight capture scripts (`remote_client.py` or synthetic test scripts such as `test_udp_sender.py`), streaming float32 PCM packets over local Wi-Fi/Ethernet. The engine dynamically spawns grid sources tied to unique `(client_ip, client_port)` addresses and reclaims them upon timeout. This mode was verified in real cross-device local area network environments with active firewalls.
4. **Mode 4: "Hunter vs. Evaders" Networked Minigame (Deliberately Shelved)**  
   A proposed multiplayer game mode utilizing the positional audio engine as a stealth mechanic. This mode was **deliberately deprioritized early and subsequently shelved outright**. This was a conscious, explicit design call: time and computational budgets were directed toward signal-processing rigor, acoustic defensibility, and sandbox stability rather than gameplay networking and state synchronization. It was not abandoned through oversight or neglect.

---

## 2. Why FFT-Based DSP Instead of Classic IIR Filtering

### 2.1 Course Syllabus and Academic Defensibility
The most critical architectural decision in the entire signal-processing pipeline was the replacement of the original wall-muffling filter. The original implementation relied on SciPy's Butterworth IIR filter cascades via second-order sections (`scipy.signal.sosfilt`). 

In the BUET Signals and Systems syllabus, the **Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT)** are thoroughly covered early in the term, whereas continuous Laplace and discrete **Z-transform filter synthesis (Bilinear Transform, Butterworth poles, IIR stability)** appear roughly three to four weeks later. Defending an IIR filter in an academic project viva before the relevant transform theory had been taught was unacceptable.

The filtering architecture was therefore completely redesigned around a direct, per-block frequency-domain multiplication via NumPy's Real Fast Fourier Transform (`np.fft.rfft` / `np.fft.irfft`). This was a **scope-appropriateness and academic defensibility decision**, chosen to ensure every mathematical operation mapped directly to syllabus concepts.

### 2.2 Filter Implementation: Per-Block Raised-Cosine Windowing
Rather than an ideal "brick-wall" lowpass filter (which causes severe time-domain sinc-ringing / Gibbs phenomenon), `dsp_engine.py` implements a raised-cosine (Hann-shaped) transition band:

$$\text{gain}(f) = \begin{cases} 
1.0 & f \le f_{\text{low}} \\
0.5 \left(1.0 + \cos\left(\pi \frac{f - f_{\text{low}}}{f_{\text{high}} - f_{\text{low}}}\right)\right) & f_{\text{low}} < f < f_{\text{high}} \\
0.0 & f \ge f_{\text{high}}
\end{cases}$$

where:
- $f_{\text{low}} = \max(0.0, f_{\text{cutoff}} - \frac{\text{\_TRANSITION\_BW\_HZ}}{2})$
- $f_{\text{high}} = f_{\text{cutoff}} + \frac{\text{\_TRANSITION\_BW\_HZ}}{2}$
- $\text{\_TRANSITION\_BW\_HZ} = 500.0\text{ Hz}$

### 2.3 Justification for Omitting Overlap-Add / Overlap-Save
`dsp_engine.py` applies this FFT filter independently to each block of $N = 1024$ samples **without overlap-add or overlap-save buffering**. The docstring of `dsp_engine.py` explicitly details the justification:

1. **Short Effective Impulse Response:**  
   The smooth frequency-domain rolloff ($\Delta f = 500\text{ Hz}$) produces rapid time-domain decay. The effective length of the filter's impulse response is approximately:
   $$L_{\text{eff}} \approx \frac{2}{\Delta f} \cdot f_s = \frac{2}{500} \cdot 44100 \approx 176\text{ samples}$$
   Relative to the block length of $N = 1024$ samples ($~23.2\text{ ms}$ at $44.1\text{ kHz}$), $L_{\text{eff}}$ is small ($\approx 17\%$ of the block). Consequently, circular-convolution time-domain aliasing (wrap-around from sample $N-1$ to $0$) affects only a small boundary region and remains imperceptible in real-time listening.
2. **Gradual Parameter Trajectories:**  
   Cutoff frequencies change only when listener, sources, or walls are manipulated between UI frames, never jumping abruptly from block to block during steady-state playback.
3. **Conceptual Clarity Over Broadcast Purity:**  
   The project prioritizes clarity of physical and mathematical concepts over mastering-grade audio processing.
4. **State Machine Simplicity:**  
   Eliminating inter-block overlap-add state buffers radically reduced the bug surface area, avoided state-synchronization hazards during source repositioning, and resulted in a clean implementation easily explained in an academic examination.

---

## 3. The Acoustic Model: Three Independent Arrival Families

### 3.1 Deficiencies of the Legacy Single-Reflection ($k=2$) Model
The original acoustic model routed audio through Yen's $k$-shortest path algorithm with $k=2$, attempting to represent a direct path ($k=1$) and a single averaged reflection ($k=2$). This approach suffered from two fatal flaws:
1. **Total Occlusion Discontinuity:** When walls completely enclosed a source or listener, pathfinding returned zero paths. The engine handled this with an artificial constant floor gain (`_TOTAL_OCCLUSION_FLOOR_GAIN = 0.05`). Moving a single wall block to seal or open an enclosure caused audio to snap violently between the floor gain and routed audio.
2. **Physical Thinness:** Averaging room reflections into a single secondary path failed to capture the spatial perception of enclosed geometry.

### 3.2 Deliberate Rejection of Generic Reverb Tails
A standard statistical artificial reverberation algorithm (such as a Freeverb comb/allpass network or Feedback Delay Network) was explicitly evaluated and **rejected**. 

A generic reverb tail adds diffuse decay energy detached from specific geometry. The primary goal of this sandbox is **geometric explicability**: users visually draw walls on a grid and expect an intuitive, audible 1:1 correspondence between physical surfaces and spatial sound arrivals. The engine therefore models discrete, geometrically grounded acoustic arrivals originating from identifiable physical walls.

### 3.3 The Three Independent Arrival Families
In the current architecture, each audio block is decomposed into three independent acoustic arrival families, separately filtered, gained, delayed, panned, and summed:

$$\mathbf{y}[n] = \mathbf{y}_A[n] + \mathbf{y}_B[n] + \sum_{i=0}^{M-1} \mathbf{y}_{C, i}[n]$$

#### Family A: Primary Routed Arrival (Diffracted / Line-of-Sight)
- **Role:** Models the shortest unobstructed or corner-diffracted acoustic path bending around obstacles.
- **Routing:** Computed via Yen's $k$-shortest path algorithm (`find_k_paths(..., k=1)`) with 8-directional A* search.
- **Lowpass Cutoff:** Driven by path corner count:
  $$f_{\text{cutoff, A}} = \max\left(\text{\_MIN\_CUTOFF\_HZ}, \text{\_BASE\_CUTOFF\_HZ} - \text{corners} \cdot \text{\_CUTOFF\_DROP\_PER\_CORNER\_HZ}\right)$$
  where $\text{\_BASE\_CUTOFF\_HZ} = 4000.0\text{ Hz}$, $\text{\_CUTOFF\_DROP\_PER\_CORNER\_HZ} = 900.0\text{ Hz}$, and $\text{\_MIN\_CUTOFF\_HZ} = 300.0\text{ Hz}$.
- **Line-of-Sight Bypass:** When direct line-of-sight is clear (`has_line_of_sight(...)` is `True`), the corner-penalty calculation is bypassed, returning $\text{\_BASE\_CUTOFF\_HZ} = 4000.0\text{ Hz}$. This prevents diagonal paths on a discrete grid from incurring artificial corner-muffling penalties due to raster discretization.
- **Distance Gain:** Inverse-distance model:
  $$g_A = \frac{1.0}{1.0 + 0.08 \cdot d_{\text{path}}}$$
- **Panning:** Linear horizontal offset panning based on straight-line source-to-listener separation:
  $$\Delta x = x_{\text{source}} - x_{\text{listener}}, \quad \text{pan} = \text{clip}\left(\frac{\Delta x}{20.0}, -1.0, 1.0\right)$$
- **Occlusion Behavior:** When no routed path exists (source fully enclosed), Family A produces zero output.

#### Family B: Through-Wall Transmission (Direct Penetration)
- **Role:** Models sound energy penetrating directly through solid barriers, completely eliminating the total occlusion volume snapping problem.
- **Routing:** Bresenham line walk from source to listener via `find_transmission_path`. Returns an ordered list of traversed wall cells and their material gains.
- **Lowpass Cutoff:** Steeper than Family A to model high-frequency barrier absorption:
  $$f_{\text{cutoff, B}} = \max\left(\text{\_MIN\_CUTOFF\_HZ}, \text{\_TRANSMISSION\_BASE\_CUTOFF\_HZ} - N_{\text{walls}} \cdot \text{\_TRANSMISSION\_CUTOFF\_DROP\_PER\_WALL\_HZ}\right)$$
  where $\text{\_TRANSMISSION\_BASE\_CUTOFF\_HZ} = 1500.0\text{ Hz}$ and $\text{\_TRANSMISSION\_CUTOFF\_DROP\_PER\_WALL\_HZ} = 500.0\text{ Hz}$.
- **Gain:** Distance attenuation combined with the product of wall transmission factors:
  $$g_B = \left(\frac{1.0}{1.0 + 0.08 \cdot d_{\text{Euclid}}}\right) \prod_{w \in \text{crossed}} (1.0 - \text{material\_gain}_w)$$
  High-reflectivity materials (Concrete, gain 0.90) yield a transmission factor of $1.0 - 0.90 = 0.10$, whereas soft materials (Curtains, gain 0.25) yield $1.0 - 0.25 = 0.75$.
- **Panning:** Along the direct straight line between source and listener.
- **Independence:** Contributes additively in both open and totally-occluded topologies.

#### Family C: Discrete Multi-Wall Echoes (Specular Reflections)
- **Role:** Models up to $M = 4$ (`_MAX_SIMULTANEOUS_ECHOES`) discrete early reflections off physical boundaries.
- **Candidate Identification:** `find_reflector_candidates` casts 12 evenly-spaced radial rays from the source out to a maximum radius of 20 grid units. A hit wall cell qualifies if both source-to-wall and wall-to-listener lines of sight are unobstructed.
- **Ranking:** Candidates are ranked by predicted loudness:
  $$L(c) = \left(\frac{1.0}{1.0 + 0.08 \cdot (2.0 \cdot d_{\text{wall}})}\right) \cdot \text{material\_gain}$$
  The top $\le 4$ candidates are selected.
- **Lowpass Cutoff:** Muffled proportionally to surface absorption:
  $$f_{\text{cutoff, C}} = \max\left(\text{\_MIN\_CUTOFF\_HZ}, \text{\_ECHO\_BASE\_CUTOFF\_HZ} - (1.0 - \text{material\_gain}) \cdot \text{\_ECHO\_CUTOFF\_MUFFLE\_RANGE\_HZ}\right)$$
  where $\text{\_ECHO\_BASE\_CUTOFF\_HZ} = 3500.0\text{ Hz}$ and $\text{\_ECHO\_CUTOFF\_MUFFLE\_RANGE\_HZ} = 3000.0\text{ Hz}$.
- **Gain:** Scaled by round-trip distance and material reflectivity:
  $$g_C = \left(\frac{1.0}{1.0 + 0.08 \cdot 2.0 \cdot d_{\text{wall}}}\right) \cdot \text{material\_gain}$$
- **Delay Line:** Each active echo slot maintains an independent circular ring buffer (`echo_delay_buf_i`, `echo_delay_write_pos_i` in `filter_state`). Delay in samples is computed from round-trip travel:
  $$\text{delay\_samples} = \text{clip}\left(\text{round}(2.0 \cdot d_{\text{wall}} \cdot \text{\_SAMPLES\_PER\_GRID\_UNIT}), 0, \text{\_MAX\_DELAY\_SAMPLES}\right)$$
  where $\text{\_SAMPLES\_PER\_GRID\_UNIT} = 40\text{ samples}$ ($\approx 0.907\text{ ms}$ per grid unit at $44.1\text{ kHz}$) and $\text{\_MAX\_DELAY\_SAMPLES} = 8000\text{ samples}$ ($\approx 181.4\text{ ms}$).
- **Panning:** Each reflection is panned independently toward its reflecting wall cell: $\Delta x = x_{\text{wall}} - x_{\text{listener}}$.
- **Total Occlusion Behavior:** In a fully walled-off room, Family A yields 0, while Family B (through-wall bleed) and Family C (reflections off interior walls) generate continuous, physically grounded acoustic output. If an entity is completely enclosed with zero valid reflectors and dense barriers, audio drops smoothly to silence without artificial gain jumps.

---

## 4. Geometry Caching and the Threading Model

### 4.1 Edit-Time Caching Principle
The audio callback executes on high-priority PortAudio OS threads at 44.1 kHz with a block size of 1024 frames ($~23.2\text{ ms}$ budget). Calling graph pathfinding (A*, Yen's), radial ray fans, and Bresenham rasterizations on every block iteration for multiple active sources would consume unpredictable CPU cycles and risk real-time audio buffer dropouts.

In `dsp_engine.py`, geometry calculations for Family B and Family C are cached inside `filter_state`:
- `_geometry_cache_is_stale()` verifies whether `source_pos`, `listener_pos`, or `walls` (stored as an immutable `frozenset`) have changed since the prior block.
- If and only if an edit has occurred, `find_reflector_candidates` and `find_transmission_path` are re-evaluated, and a new snapshot is stored via `_geometry_cache_store_snapshot()`.

**Critical Load-Bearing Invariant:**  
This caching optimization is valid **strictly because entities remain stationary during listening demonstrations**. Users reposition sources or edit walls, and then listen to the steady-state acoustic response. If continuous, real-time trajectory motion (such as physics simulations or dragging sources during active playback) were introduced, geometric search would run every frame, requiring spatial partitioning structures or an asynchronous background geometry worker.

### 4.2 Thread Isolation and Render-Side Recomputation
`shared_state.py` establishes the strict rule that it is the **sole communication channel** between the main UI thread (Pygame) and the audio callback thread (PortAudio):
- The audio callback reads state exclusively via `SharedState.get_snapshot()`, which takes a brief lock, creates a deep copy of sources and wall positions, and releases the lock immediately.
- The `filter_state` dictionary is owned privately by `AudioEngine._filter_states` and mutated exclusively inside the audio callback. It is never exposed across threads.

Consequently, `render_engine.py`'s path-drawing routine `_draw_source_paths()` **does not read the audio thread's cache**. Instead, it independently recomputes `find_k_paths(..., k=1)`, `find_transmission_path()`, and `find_reflector_candidates()` directly on the render thread using UI getters (`get_listener_pos()`, `get_sources()`, `get_walls()`, `get_wall_gains()`).

**Deliberate Architectural Trade-Off:**  
This duplicates cheap, stateless geometric computation (taking $<1\text{ ms}$ on a $40 \times 24$ grid) in exchange for:
1. Zero thread contention or lock synchronization between UI rendering and audio processing.
2. Complete elimination of race conditions or memory tears on delay-line buffers.
3. Total architectural independence of the UI visualization from the audio playback engine.

### 4.3 Visualization Specifications
Rendered paths reflect the three arrival families using dedicated color tokens:
- **Family A (Primary Routed Path):** Solid teal lines (`COL_PATH_PRIMARY = (50, 200, 180)`), width 2 px.
- **Family B (Through-Wall Transmission):** Dashed orange line (`COL_PATH_TRANSMISSION = (255, 140, 0)`), drawn via `_draw_dashed_polyline()`.
- **Family C (Discrete Echoes):** Thin magenta rays and circular wall bounce markers (`COL_PATH_ECHO = (200, 100, 220)`), ray width 1 px, circle radius 4 px.

---

## 5. The Multi-Material Wall System

### 5.1 Materials and Acoustic Parameters
Walls support three distinct material classes with pre-calibrated acoustic coefficients defined in `shared_state.py` and visualized via sidebar swatches:

| Material | Class | Reflectivity Gain ($\alpha_r$) | Display Color | RGB | Acoustic Behavior |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Hard** | Concrete / Brick | `0.90` | Light Grey | `(160, 165, 175)` | High reflection, low transmission ($1 - 0.90 = 0.10$), crisp echo |
| **Medium** | Wood / Plaster | `0.60` | Warm Tan | `(185, 155, 110)` | Balanced reflection and transmission ($1 - 0.60 = 0.40$), default material |
| **Soft** | Curtain / Carpet | `0.25` | Teal Grey | `(80, 155, 145)` | Low reflection, high transmission ($1 - 0.25 = 0.75$), severe HF absorption |

`DEFAULT_WALL_GAIN` defaults to `WALL_GAIN_MEDIUM` (`0.60`).

### 5.2 Interaction with Arrival Families
The material gain scalar dynamically shapes the acoustic output:
- **Through-Wall Transmission (Family B):** Multiplied by $(1.0 - \text{material\_gain})$. Penetrating soft curtains preserves significant acoustic energy, while concrete walls attenuate transmission by 90% per wall crossed.
- **Discrete Echoes (Family C):** Multiplied directly into echo gain ($g_C \propto \text{material\_gain}$). Furthermore, the lowpass cutoff decreases with absorption:
  $$\Delta f_{\text{muffle}} = (1.0 - \text{material\_gain}) \cdot 3000\text{ Hz}$$
  Concrete reflections retain crisp high frequencies ($f_{\text{cutoff}} \approx 3200\text{ Hz}$), whereas soft curtains heavily damp high frequencies down toward the floor cutoff ($f_{\text{cutoff}} \approx 1250\text{ Hz}$).

---

## 6. Known Limitations and Deliberately Accepted Simplifications

### 6.1 Deliberately Accepted Simplifications in Active Code
1. **Ray-Fan Search Without Angle-of-Incidence Checks:**  
   `find_reflector_candidates()` casts a 12-ray radial fan and accepts any wall cell with clear line of sight to both source and listener. It does **not calculate wall surface normals, tangent planes, or Snell's law angle of incidence/reflection ($\theta_i = \theta_r$)**. In a discrete grid sandbox, computing continuous surface normals on single-cell walls introduces substantial edge-case complexity. Treating qualifying line-of-sight wall cells as diffuse/specular reflector candidates is an intentional, acknowledged simplification.
2. **Straight-Line-Only Transmission:**  
   `find_transmission_path()` walks only the direct Bresenham Euclidean segment between source and listener. It does not perform an exhaustive search for alternative, non-straight through-wall paths through thinner partition sections.
3. **Uninterpolated Delay-Line Readhead:**  
   When entity movement causes `delay_samples` to change between blocks in `_delay_line_process()`, the write head reads from the new index immediately without fractional delay Hermite interpolation or cross-fading. This can produce minor clicking artifacts during repositioning, which was accepted because the sandbox is operated in static listening configurations.

### 6.2 Resolved Historical Issues and Evolved Architecture
1. **The Corner-Material Lookup Ambiguity:**  
   *Historical Context:* Following the merge of the teammate's multi-material wall feature (commit `1cddf97`), `dsp_engine.py` included a helper `_corner_material_gain` that probed the 8 grid neighbours of a path corner in fixed priority order: N, S, W, E, NW, NE, SW, SE. If a corner cell touched two different wall materials simultaneously, whichever neighbour appeared first in the tuple (almost always North) won the tie-break, regardless of which wall the sound was bending around.  
   *Evolution:* The code documented this as a known simplification. In Stage 6a, this ambiguity was rendered entirely **moot**: Family A was decoupled from wall reflectivity gains (reflectivity belongs to reflection paths, not diffracted paths), and the old $k=2$ single-reflection path was deleted. In the current engine, Family C queries the exact wall cell struck by the radial ray (`wall_gains.get(hit)`), and Family B queries the exact wall cells crossed by the ray (`wall_gains.get((x, y))`), eliminating corner-material tie-breaking.
2. **The Reflected-Arrival Panning Lookahead Bug:**  
   *Historical Context:* The legacy $k=2$ code contained an acknowledged bug: while its docstring stated panning used the "final segment direction", the implementation averaged lookahead steps with a division by the lookahead count that caused erratic oversensitivity on short paths.  
   *Resolution:* Rather than applying a targeted patch to flawed code, the issue was deferred and completely resolved when the entire legacy $k=2$ subsystem was excised in Stage 6a.
3. **The Multi-Source Tanh Soft-Clipper Conditional:**  
   In `audio_engine.py`, the master mix output is passed through a hyperbolic tangent soft-clipper to prevent digital clipping:
   ```python
   # ---- Soft-clip to prevent distortion when sources overlap ----
   if active_count > 1:
       mix = np.tanh(mix)
   ```
   During the teammate merge, this check had been silently made unconditional (`mix = np.tanh(mix)`), compressing the dynamic range of single sources. A subsequent audit caught the regression, restoring the conditional `if active_count > 1:`. Single sources pass through completely uncompressed with linear fidelity.

---

## 7. Testing and Verification Philosophy

### 7.1 The Verbatim-Audit Convention
A foundational rule of this codebase is the **Verbatim-Audit Convention**. Early in the project's development, an AI-assisted architectural audit relying on natural-language descriptive summaries hallucinated function signatures, non-existent parameters, and incorrect line numbers. 

As a permanent standing rule:
- Natural-language summaries of code modifications are never accepted as ground truth.
- Every architectural stage must be verified by a read-only audit inspecting raw code pastes, verifying exact function signatures, and proving line-by-line compliance against specification requirements.

### 7.2 Stage-Splitting Engineering Methodology
Major system overhauls are rigorously pre-split into small, isolated stages (e.g., Stages 1 through 6, with sub-stages such as Stage 5 Parts 1, 2, and 3, and Stage 6a, 6b, and 6c). Each stage:
- Fits comfortably within a single interaction session budget.
- Focuses on a single architectural objective (e.g., data structures, audio-thread wiring, visualization, or dead-code elimination).
- Leaves the codebase in a fully compiling, importable, passing state with zero broken tests.

### 7.3 Test Suite Structure and Conventions
The test suite avoids heavyweight testing dependencies (such as pytest or unittest test runners), utilizing standalone scripts executed directly with `python <test_file>.py`. Every script prints clean, human-readable `[PASS]` or `[FAIL]` assertions:

1. **`test_dsp_engine.py` (24 Test Functions, 117 Assertions)**  
   Verifies `process_block()` using synthetic sine waves (`_make_sine`) and white noise (`_make_noise`). Checks high-frequency spectral energy (`_hf_energy`), line-of-sight cutoff bypass, corner attenuation, total occlusion behavior, transmission attenuation and cutoff filtering, multi-slot circular delay lines, echo candidate count shrinkage and slot buffer flushing, material gain scaling, and geometry cache staleness tracking.
2. **`test_pathfinding.py` (16 Test Functions)**  
   Tests 8-directional Yen's A* routing, Bresenham line-of-sight checks, diagonal and orthogonal path distance metrics, corner counting, wall detour routing, fully walled-off total occlusion, reflector candidate radial ray fan search, candidate loudness sorting, search radius boundaries, and transmission path wall intersection lists.
3. **`test_viz_engine.py`**  
   Validates visualization DSP routines: waveform downsampling (`downsample_for_width`), display buffer scaling (`compute_display_samples`), FFT magnitude spectrum binning (`compute_spectrum`), peak frequency accuracy against pure tones, and graceful handling of empty/None buffers.
4. **`test_udp_sender.py`**  
   Diagnostic socket harness simulating two concurrent streaming clients sending float32 PCM packets to localhost:50007, verifying network source auto-spawning, positional DSP playback, and timeout garbage collection.

---

## 8. Development Process Note

### 8.1 Model Orchestration Workflow
Development followed a structured model-specialization workflow inside the Google Antigravity IDE:
- **Claude Opus (Opus 4.6 Thinking):** Utilized for high-level architecture planning, mathematical modeling, algorithm design (decoupling Yen's algorithm, formulating arrival family interactions, circular delay line buffer management), and prompt decomposition into verifiable stages.
- **Gemini Flash:** Utilized for bounded implementations, mechanical test suite adaptations, docstring synchronization, and strict read-only verbatim verification audits.

### 8.2 Rationale-Dense Code Documentation
Because development occurred across multiple decoupled stages and context windows, the codebase adopted an explicit policy of embedding deep technical rationale directly into module and function docstrings. Every constant, architectural boundary, and rejected design alternative is documented at the point of implementation, ensuring that the code itself serves as the authoritative, permanent record of engineering intent.
