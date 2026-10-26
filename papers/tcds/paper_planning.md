# Paper planning — IEEE TCDS article

**Status:** master planning document (24 Oct 2026). It supersedes `narrative/narrative_outline.pdf`. Everything needed to write the paper is here: the storyline, a section-by-section blueprint, the vocabulary, the full technical record, every number with its source, and a map of all references. The draft (`draft/main.tex`, packaged as `draft/overleaf_import/paper_draft_ver_6.zip`, with `draft/supplement.tex` as the supplementary material) is written from this document; if the two disagree, this document is updated first.

**Title (since ver_6):** *From Coupled to Estimation-Coupled Oscillators: Grounding Attachment Dynamics in a Robot Dyad* (earlier working title: *Private Needs, Shared Rhythms*)

Alternatives:
- *What a Robot Needs to Coordinate With a Partner It Cannot Read: Partner Models in a Coupled-Oscillator Model of Attachment*
- *Decoupling the Dyad: Partner Models for Attachment-Like Coordination Between Robots*

**Framing (decided):** robotics first, without neglecting attachment theory. Attachment is the test case and the source of the theory. The take-home is a design lesson for robots, and the paper also makes a substantive point about attachment.

---

## A. Narrative since ver_6: grounding the coupling (supersedes §1 and the opening of §3 where they differ)

The npj draft is no longer a separate paper; its content is part of this article.

**Logline.** A robot can couple to a partner only through what it senses. The coupling terms of the LM model (physical and emotional distance) are computed from both partners' needs, so they are ungrounded. We ground them in three steps, and test robots coupled only through estimation.

**Spine.**
1. **What the coupling does** (Section III, npj analysis reproduced; Fig. 2 from `canonical.py`).
   - Ambivalence: weak ambivalence keeps the needs synchronised (median r 0.90 at εAm = 0.1); stronger ambivalence anti-synchronises them (−0.96 at 0.7) and slows them.
   - Inhomogeneous steady state: born at a saddle-node at εAm = b = 0.5, stable from about 0.75; oscillation ends at about 0.76 from random starts.
   - Avoidance: anti-synchronised at first (−0.94 at 0.1), then synchronised (0.67 at 0.6), then a shared exploration fixed point, stable from about 0.69.
   - Plane: 81% of parameter pairs oscillate, 5% settle, 14% escape; 10% show irregular oscillation (disorganisation candidate).
2. **What it looks like from outside** (Section IV).
   - Simulated SSP (`ssp_sim.py`): time near the caregiver 50 / 38 / 18 / 53 / 100%, χ² = 14,676, N = 50,000; npj reported 52 / 41 / 18 / 52 / 99.9%.
   - Physical MiRo-e robots (npj manuscript, figures in S4): 54 / 41 / 43%, χ² = 99.38, N = 29,743.
3. **Learning the coupling terms** from distance and calls in the same agent-based SSP; accurate offline.
4. **Estimation-coupled robots** (R1–R5 as before): accurate estimates are not enough; anchoring is.
5. **Meaning:** coupling must be grounded *and* anchored; the regimes correspond to attachment patterns (irregular oscillation ↔ disorganisation); Bowlby's fitted working models.

**Model forms (stated in the paper).**
- **Canonical (Section III; npj code `attachment_model_inference`):**
  - $\dot x=x-2x^3-y$;
  - $\dot y_{c/p}=\mu(b x \mp \varepsilon_{Am}\Phi_p - \varepsilon_{Av}\Phi_e)$;
  - $\Phi_p=\tfrac12(y_c+y_p)+y_c$, $\Phi_e=k_0(\tfrac12(y_c-y_p)+y_c-\delta)$;
  - $b=0.5$, $\mu=0.1$.
- **Robot controllers (distance form, Eq. 5):** $b'=0.7$. Same phase relations as the canonical model; escapes where the canonical model settles (S2, Fig. S1).
- **Training simulator:** opposite sign of the $k_0\delta$ offset (S1).

**Supplement map (ver_6):**
- S1 training;
- S2 model forms and escape;
- S3 simulated SSP;
- S4 physical robots (npj Figs. 10–12);
- S5 estimates;
- S6 robot behaviour;
- S7 observers;
- S8 runs and code names.

**Open:**
- confirm μ = 0.1 and the δ sign;
- npj's "chaotic band" reproduces only as irregular, non-locked oscillation near the transitions and at weak coupling, not as broad chaos;
- physical-robot recordings are not in the repo;
- optional pulse-corrected physical runs in `attachment_model_inference/src/save_records`.

## 0. TCDS constraints and page budget (since ver_5)

TCDS rules for regular papers (from the author guidelines):
- The abstract must be **150–250 words**.
- Each page beyond **10** pays a mandatory overlength charge of $200. The count includes references and author biographies but **excludes supplementary material**. The maximum length with charges is 15 pages.
- The $110 per page charge is voluntary and **will be declined**.

**Decision:** no extra payments. The paper must end by page 10, and everything that does not fit goes into the supplementary material (`draft/supplement.tex`).

Budget, measured in the IEEEtran journal layout, where a page of pure text holds about 1050 words:

| Part | Budget | ver_5 |
|---|---|---|
| Abstract | about 200 words (150–250) | 224 words |
| Running text | about 4,900 words | 4,350 words (introduction 660, background 400, model 230, private models 500, methods 580, results 1,350, discussion 510, conclusion 120) |
| Figures and tables | about 2 pages | Figs. 1–7 and Table I |
| References and biographies | about 1.3 pages | 33 references cited in the paper; 4 biography placeholders |
| **Paper ends on** | **≤ page 10** (aim about 9.3) | **page 9**, about 1.5 pages of headroom |

Word targets per section:

| Section | Words |
|---|---|
| Introduction | 650 |
| Background | 450 |
| Model | 280 |
| Private working models | 520 |
| Methods | 600 |
| Results | 1,600 (R1 220, R2 360, R3 250, R4 480, R5 230) |
| Discussion | 560 |
| Conclusion | 110 |

The headroom absorbs:
- about six references still to be sourced (the `\citeneeded` placeholders);
- the real biographies;
- IEEE typesetting;
- additions requested in review.

**Enforced by `make_overleaf_zip.sh`.** It refuses to package a draft if the label `paper:end` (after the last biography) lands beyond page 10, if the abstract is outside 150–250 words (`texcount`), or if a supplementary section, figure or table cited in the paper is missing from the supplement. It also test-compiles the submission version (`\draftmarkersfalse`).

## 1. Logline and thesis

**Logline.** A robot that has to coordinate with a partner whose needs it cannot read needs a working model of that partner that is *anchored to what it perceives*. Without anchoring, the robot keeps acting on a relationship that is not taking place.

**Thesis.** In a coupled-oscillator model of attachment, coordination between two robots depends on what each robot's partner model does:
- With shared needs, the dyad coordinates: synchrony under avoidance, anti-synchrony under ambivalence.
- With private partner models that run on the robot's own learned estimates, each robot keeps believing in a coordinated relationship while the real rhythms drift apart.
- Coordination returns when the partner model follows a reference model of the partner *and* the robot perceives the partner. The two robots' rhythms then stop being detuned.
- The difference is visible in behaviour (whether the caregiver answers the child's calls), and an observer can read the attachment parameters from it.

**One sentence per result** (R1–R5 below):
1. **R1.** With shared needs the dyad coordinates, and the robots show it: the caregiver answers every call under avoidance and about a quarter of them under ambivalence.
2. **R2.** With private models each robot believes in an (anti-)synchronised relationship while the real rhythms drift apart, because the two private models are detuned.
3. **R3.** The synchrony seen from identical starting needs is an unstable artefact of the symmetric start.
4. **R4.** Anchoring the partner model (a reference model plus perception) removes the detuning: the model tracks the partner and the ambivalent anti-synchrony returns, at a cost in stability.
5. **R5.** Behaviour carries the internal regime: reciprocity separates coordinated from drifting dyads, and the attachment parameters can be recovered from behaviour, mostly from the caregiver's responsiveness.

---

## 2. Why the previous draft does not work (and what changes)

| Problem in `paper_draft_ver_2` | Change |
|---|---|
| Organised by method and controller variant, not by question; six variants listed up front | Organised by the five results R1–R5; variants introduced as a 2×2 design with one purpose |
| The two-factor logic (what the estimator perceives × how the partner model is propagated) is hidden | Fig. 2 marks the two factors and Section IV-C names the four designs; descriptive names replace code names (no separate design table, by decision of 24 Oct 2026) |
| Results paragraphs pack 2–4 numbers per sentence; no topic sentence; no "so what" | Each results subsection: claim → figure → mechanism → meaning; at most 2 numbers per sentence |
| Robots appear late, as an "evaluation" | Robot behaviour opens the results (R1) and closes them (R5) |
| The climax (anchoring restores coordination) is a bullet in Section VI | R4 is its own subsection with its own figure (`fig_anchoring`) |
| The identical-start finding reads as a correction to an earlier version | R3 is a result in its own right and a methodological lesson |
| Bullet lists in introduction, results and discussion | Prose; lists only in tables and Methods |
| Terminology drifts | Glossary (Section 5) used verbatim |
| References decorate | Each reference has a job in the argument (Section 8) |

---

## 3. The story

### 3.1 The problem (hook)
Social and companion robots are being built to keep company with people over long periods, in homes, care settings and therapy. Relationships with them can help and can harm (Prescott & Robillard 2021; Collins et al. 2013; Sharkey & Sharkey 2010). To relate well, a robot has to notice when a person seeks contact or withdraws and respond in kind. The person's needs are not observable: they are interoceptive, sensed only from within (Craig 2002). A robot can see only behaviour, so it must act on a *model* of its partner, as affect-aware assistive systems already do when they infer a user's affective identity (BayesACT; Robillard & Hoey 2018). Designers increasingly aim for robots whose interactions *resonate*, entraining with the user's rhythms (Lomas et al. 2022). The question is what a robot's model of its partner must do for such coordination to arise.

### 3.2 The test case and its theory
Attachment is the prototype of a relationship regulated by both partners' behaviour.
- **Bowlby (1969/1982)** described the child–mother pair as a "dynamic equilibrium". The child's attachment and exploration, and the mother's caretaking and other activity, elicit and inhibit one another, keeping distance "within certain stable limits"; "in a happy couple these four classes of behaviour occur and progress together in harmony".
- **The SSP.** Ainsworth's strange situation (Ainsworth et al. 1978) turned this into a laboratory measure. Continuous recordings show quasi-periodic excursions and returns (Chow et al. 2018; Prince et al. 2021).
- **Biobehavioural synchrony.** Human attachments are characterised by biobehavioural synchrony (Feldman 2017).
- **Dimensional view.** Individual differences vary continuously along avoidance and ambivalence (Fraley & Spieker 2003; Gagliardi 2021, 2022).

**Bowlby's working models.** Bowlby also proposed that each partner acts on *working models*. To plan, an animal needs "something more like a working model of its environment". The child builds working models of "how … his mother … may be expected to behave, how he himself may be expected to behave, and how each interacts with all the others", and "only gradually … does his working model of her become fitted for its role" (Bowlby 1969, Pelican edition; page numbers to be checked in the 1982 edition).

### 3.3 The gap
Our coupled-oscillator model (Htet et al. 2024, "LM") captured the dyad as two motivational oscillators (Jimenez-Rodriguez et al. 2020) coupled by an attractive avoidance term and a repulsive ambivalence term (cf. Dixit et al. 2019). It produced synchronised, anti-synchronised and steady-state regimes resembling secure, avoidant and ambivalent patterns, and two MiRo robots running it showed matching behaviour. But one controller integrated both oscillators, so each agent's coupling read the other's need directly.

The closest computational precedent shares the shortcut: in Gagliardi's (2022) agent-based SSP, "each agent can perceive the other's need level". Dynamic models of attachment in this journal's lineage (Stevens & Zhang 2009) treat the caregiver as parameters rather than as an agent with a model of the child. **No model so far asks what happens when each partner has only its own working model of the other** — the situation Bowlby described, and the one every robot is in.

### 3.4 The approach
Two simulated MiRo robots (Prescott et al. 2017; Mitchinson & Prescott 2016), child and caregiver, are placed in a strange-situation arena in Gazebo.
- **Private working model.** Each robot holds its own copy of the dyad: its own oscillator, a model of its partner, and the coupling between them.
- **Learned estimate.** Its estimate of how the partner pulls on it comes from neural networks trained in an agent-based strange situation on signals a robot can sense (distance and calls).
- **Two design factors:**
  - what the estimate *perceives*: nothing (the internal model's need gap and silent calls) vs. the sensed distance and the partner's calls;
  - how the partner model is *propagated*: by the robot's own learned estimate vs. as a *reference* that follows the standard model.
- **Controls:** the shared-needs (coupled) dyad, and a robot with no partner model at all.

All conditions are run from random starting needs, offline (thousands of starts) and on the robots (577 runs, each validated against its controller).

### 3.5 What we found — five beats

**R1 — Coordination with shared needs, and what it looks like on robots.**
- **Claim:** with shared needs the dyad coordinates from any start: synchrony under avoidance, anti-synchrony under ambivalence.
- **Evidence:**
  - offline from 40 random starts, r = 0.79–0.94 (εAv = 0.1–0.2, 180 s) and −0.94 to −0.98 (εAm = 0.1–0.2), reaching ±1.00 after transients;
  - Gazebo +0.94 and −0.95 at ε = 0.2.
- **What a robot sees (behaviour):** under avoidance the caregiver answers every one of the child's calls (reciprocity 1.00); under ambivalence its calls alternate with the child's and it answers 0.29 at εAm = 0.2 and 0.15 at 0.4, the pattern of an unavailable caregiver. (The 15-minute runs, Fig. long, are reported under R5.)
- **Meaning:** this is the reference — coordination is possible, and it has a behavioural signature an observer (or a robot) can see.

**R2 — Private models: belief without coordination.**
- **Claim:** when each robot runs its own partner model on its own learned estimate and perceives nothing, each robot keeps believing in a coordinated relationship while the real rhythms drift apart.
- **Evidence:**
  - realised correlation near zero in both regimes (offline 180 s: 0.10, −0.04 under avoidance; −0.38, −0.08 under ambivalence; within 0.11 of zero after transients for ε = 0.1–0.4);
  - belief anti-synchronised (−0.94 to −0.99 under ambivalence; −0.50, −0.81 under avoidance);
  - the partner model does not follow the partner (r ≈ 0.1);
  - on the robots: realised r = −0.13 against −0.93 for shared needs, and the caregiver answers 0.55 of calls against 0.25.
- **Mechanism:** the two robots' private models settle on different frequencies, because each robot's learned estimate is conditioned on its own need. Example: 24.8 vs 31.5 mHz at εAm = 0.2 offline; 6.1 mHz model–partner detuning on the robots. In coordination dynamics, detuning breaks phase locking and lets the relative phase wander (Kelso 1995). This is robust to which networks each robot uses: with each robot's own networks, realised r is 0.01 at εAv = 0.1 and 0.00 at εAm = 0.2.
- **Meaning for robots:** a partner model that is internally consistent — it "believes" in the relationship — tells the robot nothing about whether the relationship is happening.

**R3 — The identical-start illusion.**
- **Claim:** from identical starting needs the private-model dyad under avoidance looks perfectly synchronised, offline and on the robots. This is an unstable symmetric state.
- **Evidence:**
  - a starting difference of 10⁻⁷ grows until the needs separate after 704 s (εAv = 0.1) and 421 s (0.2);
  - 10⁻³ separates them after 322 and 188 s; 0.1 after 17–18 s;
  - each hundred-fold smaller perturbation adds a constant delay, the signature of exponential growth;
  - the shared-needs dyad returns to synchrony from every perturbation.
- **Mechanism:** identical starts make the child and caregiver halves of every copy equal, and avoidance acts on both alike, so the learned estimates coincide.
- **Meaning:**
  - a validation lesson for robot models of coupled dynamics: symmetric starts can hide instability, and the robots could not reveal it because their needs do not depend on behaviour;
  - the contrast with pulse-coupled oscillators, which synchronise "for almost all initial conditions" (Mirollo & Strogatz 1990), is instructive: here synchrony holds for almost *no* initial condition.

**R4 — Anchoring restores coordination (the climax).**
- **Claim:** coordination returns only when the partner model follows a reference model *and* the robot perceives its partner.
- **Evidence:** under ambivalence (pooled over εAm = 0.1–0.4, Gazebo):
  - the anchored design's partner model tracks the caregiver (r = 0.76 ± 0.11) against 0.13–0.31 in the other three designs;
  - the realised anti-synchrony returns (−0.62 vs −0.06 to −0.20; shared needs −0.93);
  - the caregiver answers 0.35 of calls, close to the shared-needs 0.25;
  - under avoidance (εAv = 0.1): tracking 0.79, realised r = 0.60.
- **Mechanism:**
  - anchoring removes the detuning — model–partner frequency difference 0.2 mHz against 3.5–6.1 mHz in the other designs; own–partner difference 1.6 mHz against 0.5 mHz with shared needs (3.7–5.6 mHz in the other private designs);
  - estimation accuracy is *not* the difference: the child's learned estimate follows the coupling of the true needs equally well in all four designs under ambivalence (r ≈ 0.85);
  - neither ingredient alone suffices: perception without a reference model, or a reference without perception, leaves the robots detuned.
- **Cost:** the anchored dyad escapes at weaker coupling (εAm ≥ 0.6, εAv ≥ 0.2), a narrower stable range.
- **Supporting result:** the estimate itself works when the child hears the calls (r = 0.97 under avoidance, 0.84 under ambivalence) and is inverted under avoidance when it does not (−0.51 to −0.80). The caregiver's estimate is poor in every design (−0.48 to 0.27).
- **Meaning for robots:** to coordinate, a robot needs both a principled model of how its partner behaves and perception that keeps its own response tied to the real partner.

**R5 — Reading attachment from behaviour.**
- **Claim:** the internal regime shows in behaviour, and an observer can recover the attachment parameters from behaviour.
- **Evidence:**
  - call reciprocity: shared-needs dyads hold 1.00 (avoidance) or ≈ 0.3 (ambivalence) for 15 minutes, while private-model dyads wander between 0.18 and 1.00;
  - random-forest observers trained only on a kinematic stand-in and tested on 176 Gazebo runs: 60% correct from the child alone, 89% from both robots (identical 88%, random 89%; chance ≈ 33%); parameter error falls from 0.18–0.19 to 0.05–0.07;
  - the caregiver's calling carries most of the information.
- **Meaning:**
  - for robots: a robot observing a dyad, or its own interaction, can read relational style from responsiveness;
  - for attachment research: it echoes the link between maternal responsiveness and infant pattern (Ainsworth et al. 1978) and the objective multimodal measures of Prince et al. (2021).

### 3.6 What it means
1. **For robot design.** A partner model must be anchored to perception, not just internally consistent. A concrete route is pulse coupling at discrete social events such as calls (Mirollo & Strogatz 1990), which fits the resonance design strategy (Lomas et al. 2022) and BayesACT-style inference of a partner's state (Robillard & Hoey 2018).
2. **For attachment theory.** Coordination is a joint achievement of *fitted* working models (Bowlby). The coupled model's regimes assume partners whose models of each other are already fitted. Testable prediction: in SSP recordings, partners' individual rhythms should be more regular than their relative phase in dyads with poorly fitted models (Chow et al. 2018; Prince et al. 2021; Feldman 2017).
3. **For method.** Test coupled robot models from varied starting states; validate every run against its controller.
4. **Ethics.** Robots that read and respond to relational style raise questions of dependence and manipulation (Sharkey & Sharkey 2010; Prescott & Robillard 2021).

### 3.7 Where the story stops (limitations as boundaries)
- **Simulation only:** robots are simulated (Gazebo) with a simulated Boolean audio channel; the physical study is LM.
- **Training mismatch:** the networks were trained in a simulator with different fast dynamics, rate and timescale (the robots move 6–10× more slowly relative to the need cycle).
- **Fixed parameters:** the attachment parameters are inputs to the networks, not estimated online.
- **Escape:** the model escapes at strong avoidance and, from random starts, at moderate avoidance; a bounded coupling is future work.
- **Run counts:** 3–5 Gazebo runs per condition; needs-level conclusions rest on offline controllers that every run reproduces.
- **Scope:** no disorganisation, no development over time (Stevens & Zhang 2009; Prescott & Jimenez-Rodriguez 2025), no humans.

---

## 4. Paper blueprint

TCDS regular-paper page limit to be checked. Target ≈ 12–14 pages including appendices.

### Abstract (≈ 200 words; skeleton with draft text)
1. **Problem:** social robots must act on models of partners whose needs they cannot observe.
2. **Test case:** attachment; our coupled-oscillator model read each agent's need directly.
3. **What we did:** private working models plus learned estimates in two simulated MiRo robots; two design factors (perception, partner-model propagation).
4. **R1–R4 in one sentence each, with one number each.**
5. **R5 and the scale:** 89% of 176 runs; 577 validated runs.
6. **Lesson:** a partner model must be anchored to perception.

### I. Introduction (6 paragraphs)
- **P1 Hook:** robots in relationships must model partners whose needs are private (Prescott & Robillard; Collins; Craig; Robillard & Hoey; Lomas).
- **P2 Test case:** attachment as dynamic equilibrium and synchrony (Bowlby; Ainsworth; Chow; Prince; Feldman); dimensional (Fraley; Gagliardi).
- **P3 Our model and its shortcut:** LM; Dixit; shared need reading; Gagliardi 2022 shares it; Bowlby's working models and "fitted".
- **P4 Question and approach:** private working models + learned estimates; the 2×2; random starts; 577 validated runs.
- **P5 Answer preview:** R1–R5 in prose.
- **P6 Contributions** (three or four, in prose) **+ relation to LM** (two sentences of prose; no comparison table) **+ roadmap.**

### II. Background (three subsections, each making an argument)
- **A. Robots that relate to people must model them.** Prescott & Robillard; Collins; Sharkey; Robillard & Hoey (BayesACT); Lomas (resonance). Placeholders: user-state estimation; theory of mind in social robots.
- **B. Attachment as coordinated motivational systems.** Bowlby (control system, set-goal, dynamic equilibrium, working models); Ainsworth (SSP; maternal responsiveness); Fraley; Gagliardi 2021; Feldman; Chow; Prince.
- **C. Dynamical and robotic models of attachment, and the shortcut they share.**
  - Petters; Stevens & Zhang; Buono et al. (`Buono2006`); Gagliardi 2022 ("perceive the other's need level"); Cañamero (imprinting robot); Wilson 2017 (huddling); LM.
  - Coordination dynamics: Kelso; Thelen & Smith; Ermentrout; Dixit; Helm; Palumbo; Mirollo & Strogatz.

### III. The coupled-oscillator dyad (short)
- **A. A motivational oscillator:** Jimenez-Rodriguez 2020; Prescott & Jimenez-Rodriguez 2025; fast–slow (Bertram & Rubin). Needs as interoceptive states (Craig).
- **B. Coupling two agents:** avoidance attractive, ambivalence repulsive (LM; Dixit); coupling signals physical/emotional (Gagliardi 2022); Eq. (analytic coupling) — the shared-needs dyad.

### IV. Private working models
- **A. The robot's copy of the dyad:** self + partner model + interaction (Bowlby's three components); Fig. 2.
- **B. Learning to estimate the partner's pull:** agent-based SSP; inputs; targets; training (details in Appendix A).
- **C. Two design factors** (Fig. 2; run counts per design in App. F): perception (blind/perceiving) × partner model (self-propagated/reference); controls (shared needs; no partner model).
- **D. Why private models can come apart:** identical copies would reproduce the shared dyad; conditioning on the owner's need individuates them.

### V. Method
- **A. Offline analysis:** random starts (40 per value), horizons, plane, stability perturbations.
- **B. Robots in Gazebo:** arena, behaviours, calls, logging; Fig. arena.
- **C. Runs and validation:** 577 runs; every run checked (replay or logged inputs).
- **D. Measures:** coordination r(x_c, x_p); belief r(x_c, x̂_p); partner tracking r(x̂_p, x_p); detuning; estimate accuracy; call reciprocity; proximity; escape.
- **E. Inverse problem:** stand-in training, Gazebo testing (details Appendix D).

### VI. Results
- **A. R1:** Fig. exemplars (coupled vs private, same start) + Fig. strips (coordination along both strips) + Fig. long (reciprocity over time).
- **B. R2:** Fig. belief (logs + long horizon); detuning mechanism (Kelso).
- **C. R3:** Fig. stability (+ identical-start line in Fig. strips).
- **D. R4:** Fig. anchoring (climax) + Fig. estimation (estimate and tracking along both strips); cost in stability.
- **E. R5:** Fig. inverse + Table inverse; reciprocity as the signature.

### VII. Discussion
- **A.** What a robot needs to coordinate (anchoring; pulse coupling at calls; resonance; BayesACT).
- **B.** What it says about attachment (fitted working models; predictions for SSP data).
- **C.** Testing robot models from varied starts.
- **D.** Relation to the physical robot study; limitations.
- **E.** Ethics.

### VIII. Conclusion (one paragraph)

### Supplementary material (`supplement.tex`; replaced the appendices in ver_5)
- **S1.** Training environment and parameters (Table S1).
- **S2.** The avoidance–ambivalence plane and why the needs escape (Fig. S1, escape threshold).
- **S3.** The learned estimates: offline accuracy (Fig. S2), estimation on the robots by design (Table S2, Fig. S3), belief after transients (Fig. S4), network assignment, a robot without a partner model.
- **S4.** Robot behaviour: arena (Fig. S5), behaviour at three conditions (Table S3), exemplar runs with paths (Fig. S6), 15-minute runs (Fig. S7), sweeps and kinematic stand-in (Fig. S8), odometry (Fig. S9).
- **S5.** Reading parameters from behaviour: training set, features, forests, confusion matrices (Fig. S10).
- **S6.** Runs, validation and code names (Table S4).
- The paper ends with acknowledgements, interests, data availability, references and four author biographies (placeholders that reserve the final space). A draft-only page listing the citations to be sourced follows the biographies and disappears with `\draftmarkersfalse`.

### Figure and table plan (as built in `paper_draft_ver_5`)

**Paper (`main.tex`, counted in the 10 pages)**

| # | Content | Generator | Section |
|---|---|---|---|
| Fig. 1 | Concept: shared needs → private working models → anchored partner models | `draft/diagrams/fig_concept.tex` | I |
| Fig. 2 | One robot's controller, compact, with the two design factors marked | `draft/diagrams/fig_architecture.tex` | IV |
| Fig. 3 | 2×2 exemplar timelines at ε = 0.2 (shared vs private × avoidance vs ambivalence) | `figures.py: fig_gazebo_timelines` | VI-A, VI-B |
| Fig. 4 | Coordination along both strips; identical start dashed; escape | `fig_strips` | VI-A–C |
| Fig. 5 | Belief without coordination: one controller log (column width) | `fig_belief_log` | VI-B |
| Fig. 6 | Instability of the identical start | `fig_stability` | VI-C |
| Fig. 7 | Anchoring: coordination, tracking, detuning, estimate accuracy, reciprocity | `fig_anchoring` | VI-D |
| Table I | Reading the parameters from behaviour | `tables/inverse.tex` | VI-E |

**Supplementary material (`supplement.tex`, not counted)**

| # | Content | Generator | Section |
|---|---|---|---|
| Table S1 | Parameters | in `supplement.tex` | S1 |
| Fig. S1 | Regimes over the plane | `fig_maps` | S2 |
| Fig. S2 | Learned estimates offline | `fig_estimator` | S3 |
| Table S2 | Estimation on the robots by design | `tables/estimation.tex` | S3 |
| Fig. S3 | Learned estimate and partner tracking along the strips, by design | `fig_estimation` | S3 |
| Fig. S4 | Belief and reality, with after-transient medians | `fig_belief` | S3 |
| Fig. S5 | The Gazebo arena | `fig_arena` | S4 |
| Table S3 | Behaviour at three conditions | `tables/gazebo_conditions.tex` | S4 |
| Fig. S6 | Exemplar runs with paths, incl. a quenched run | `fig_gazebo_regimes` | S4 |
| Fig. S7 | Reciprocity and coordination over 15 min | `fig_long` | S4 |
| Fig. S8 | Behaviour across the sweeps, with the stand-in | `fig_gazebo_sweeps` | S4 |
| Fig. S9 | Wheel odometry | `fig_odometry` | S4 |
| Fig. S10 | Reading the parameters from behaviour (confusion matrices, estimates) | `fig_inverse` | S5 |
| Table S4 | Design names, code names and run counts | in `supplement.tex` | S6 |

---

## 5. Vocabulary (use verbatim)

| Term | Meaning |
|---|---|
| need $x_i$ | fast motivational state of agent $i$; approach if $x_i>0$, explore if $x_i<0$ |
| accumulated need $y_i$ | slow variable that drives the need |
| shared-needs dyad (coupled) | one integrator holds both agents' needs; the LM model |
| working model / private copy | a robot's own copy of the dyad: its own oscillator, a model of its partner and their coupling |
| partner model $(\hat x,\hat y)$ | the partner half of a robot's working model |
| estimate (of the coupling) | the robot's learned estimate $d_p, d_e$ of how the partner pulls on it |
| blind / perceiving | the estimate is fed the internal need gap and silent calls / the sensed distance and the calls |
| self-propagated / reference | the partner model uses the robot's own estimate / follows the standard model (analytic coupling) |
| anchored | perceiving + reference |
| coordination | realised correlation $r(x_c,x_p)$ between the two robots' own needs |
| belief | $r(x_c,\hat x_p)$ inside a robot's working model |
| partner tracking | $r(\hat x_p,x_p)$, the child's model of the caregiver against the caregiver's own need |
| detuning | difference between need frequencies (model–partner, or own–partner) |
| call reciprocity | fraction of the child's calling time with a caregiver call within ±3 s |
| escape | a need or accumulated need exceeds 10 in magnitude (loss of boundedness) |
| random / identical start | starting needs $U(-1,1)$ from the run seed / $(-1,1,-1,1)$ |

**Design names and code names** (code names appear only in Appendix F):

| Paper | Code (`controller`, `distance`) | Batch |
|---|---|---|
| shared needs (coupled) | `coupled` | E2, E3, E4 |
| blind, self-propagated | `decoupled` | D0 (runs 34–53), E1s, E1r, E3, E4 |
| perceiving, self-propagated | `observed`, `gt` (odometry: `odom`) | E5a (E5b) |
| blind, reference | `baseline` | E6a |
| perceiving, reference = anchored | `baseline-observed`, `gt` | E6b |
| perceiving, no partner model | `notebook`, `gt` | E5c |

---

## 6. Technical record

### 6.1 Model
**Fast need:**

$$\dot x_i=-a(4c\,x_i^3-2x_i)-y_i$$

**Accumulated needs:**

$$\dot y_c=b\,x_c-\varepsilon_{Am}(y_c+y_p-d_p)-\varepsilon_{Av}(y_p-y_c-d_e)$$

$$\dot y_p=b\,x_p-\varepsilon_{Am}(y_c+y_p+d_p)-\varepsilon_{Av}(y_p-y_c-d_e)$$

**Analytic coupling signals** (shared-needs dyad and training targets):

$$d_p=\tfrac12(y_c+y_p)+y_c, \qquad d_e=k_0\big(\tfrac12(y_c-y_p)+y_c\big)+k_0\delta$$

**Parameters:**
- $a=c=1$, $b=0.7$, $k_0=0.8$, $\delta=0.1$.
- RK4 with $h=0.01$ model time per tick, at 30 Hz, so 1 s of robot time = 0.3 model time units. The uncoupled period is ≈ 32 s.

**Escape threshold** (symmetric state, high avoidance):

$$y^*(\varepsilon_{Av})=\big(b/(\varepsilon_{Av}k_0\,4^{1/3})\big)^{3/2}$$

It is 1.62 at 0.4 and 1.09 at 0.52.

### 6.2 Controllers (`ros/Attachment-Model/src/attachment_model_control.py`)
**Launch arguments:** `controller` (`decoupled` default | `coupled` | `observed` | `baseline` | `baseline-observed` | `notebook`), `distance` (`gt` | `odom`), `seed`, `initial` (`""` = identical, `random`, or four numbers), `batch`.
- **Private copies.** In the private-model modes each robot has its own thread with a 4-D copy. Both copies start from the same true starting needs (`utils.initial_state`).
- **Baseline modes.** The robot's own half uses the learned estimate. The partner half uses the analytic coupling evaluated on the copy, held over the tick.
- **Notebook mode.** Own need only, with the training simulator's equations:
  - Euler, $\Delta t=0.01$, $b=0.5$, $\dot x=x-2x^3-y$;
  - $\dot y = b x \mp \varepsilon_{Am}\Phi_p - \varepsilon_{Av}\Phi_e$.
- **Networks.** The caregiver's networks are used for both robots in the blind modes (as in the original controller); each robot uses its own networks in the perceiving modes.
- **Logs.** Each controller saves `controller_<child|parent|central>.npy` every tick: time, state, $d_p$, $d_e$, and either the observations or the partner-model coupling.
- **Other changes:**
  - `agents.py`: seeds `np.random` (child `seed`, caregiver `seed+1`); the child now listens to the caregiver's audio channel (`audio_fix`).
  - `experiment_timer.py`: `meta.json` gains controller, distance, seed, the initial state, batch, `wall_duration_s` and `audio_fix`.

### 6.3 Learned coupling estimates (`src/attachment_with_nn.ipynb`; weights `*_coupling_model.pth`)
- **Networks:**
  - $\Phi_p$: $4\to32\to64\to32\to1$, tanh; inputs [distance, own $x$, own $y$, $\varepsilon_{Am}$].
  - $\Phi_e$: $6\to64\to128\to64\to1$, SiLU; inputs [own evidence trace, partner call, own call, own $x$, own $y$, $\varepsilon_{Av}$].
- **Training environment (`simulateSS`):**
  - point agents in $[-1,1]^2$; the caregiver near the centre; the child heads for a random corner or approaches;
  - an agent in approach calls with probability $1-$call trace, and replies to calls;
  - traces: call $c\leftarrow0.9c+0.2v_{own}$, evidence $\epsilon\leftarrow0.9\epsilon+0.2\,v_{partner}c_{prev}$.
- **Training:**
  - 10 parameter values in $[0,1]$, $10^4$ steps, 100-step windows, 80/20 split;
  - loss: MSE to the analytic target + temporal smoothness;
  - Adam, lr $5\times10^{-4}$, batch 32, 50 epochs.
  - Only $\varepsilon=0$ used a recorded run; the rest is simulation.
- **Distance filter quirk.** The training distance input is `filter_dist`, a `conv1d` with a 20-tap exponential kernel ($\tau=10$) and left padding. As a cross-correlation it puts the *largest* weight on the *oldest* sample. Reproduced as trained in the robot controller and in `analysis/tcds/sim_agents.py`; verified against PyTorch to $1.8\times10^{-7}$.
- **Perceiving inputs in Gazebo:**
  - distance in metres × 2/2.61 (training arena units) through the same filter, every tick;
  - calls from `/<robot>/audio_simulation`, with the traces above updated every tick (0.01 model time = one training step).

### 6.4 Gazebo world and behaviours (`worlds/attachment.world`, `src/behaviors.py`)
- **Arena:** 2.61 m square centred on the caregiver's 0.3 m box at (0.325, 0.325), with an AprilTag on each face (`perception.py`, tag36h11). The caregiver sits on the box. The child spawns at (1.0, 0.325), facing it.
- **Explore:** 0.2 m/s commanded (≈ 0.12 m/s realised); random turn rate redrawn in even seconds; turn in place for 5 s if sonar < 0.105 m.
- **Approach:** steer to the visible tag (0.1 m/s, ±0.2 rad/s); stop if sonar < 0.2 m; rotate at 0.3 rad/s if no tag is visible.
- **Calls:** the child calls while approaching with a tag in view; the caregiver calls while its need is in approach (Boolean channel at 10 Hz).
- **Recording:** 10 Hz needs, actions, calls, odometry and ground truth, plus an overhead video. Physics runs at 1 kHz, capped at real time.

### 6.5 Run design (`experiments/make_runs.py`, `run_batch.py`)
**Strips:** (0,0) and εAv or εAm = 0.1–0.9 with the other at 0 (19 conditions). All runs are 180 s unless noted. Random-start seeds 1–5 give the same starts in every condition and design (paired); identical-start seeds 101–103 vary only exploration.

| Batch | Design | Start | Conditions × seeds | Runs |
|---|---|---|---|---|
| D0 (runs 34–53) | blind/self | identical | strips + 3 single, 1 each (180–300 s) | 20 |
| E1s | blind/self | identical | 19 × 3 | 57 |
| E1r | blind/self | random | 19 × 5 | 95 |
| E2 | shared needs | random (+identical) | 19 × (3+1) | 76 |
| E3 | shared + blind/self | random | 6 off-axis × 3 × 2 | 36 |
| E4 | shared + blind/self | random | (0.2,0),(0,0.2) × 2 × 2, 900 s | 8 |
| E5a / E5b | perceiving/self (gt / odom) | random | 19 × 3 each | 114 |
| E5c | no partner model (gt) | random | 19 × 3 | 57 |
| E6a | blind/reference | random | 19 × 3 | 57 |
| E6b | perceiving/reference (anchored) | random | 19 × 3 | 57 |
| **Total** | | | | **577** |

- **Execution:** 6 parallel containers (no `--net=host`); median real-time factor 0.97 (min 0.76); ≈ 6 h in total.
- **Data:** `ros/Attachment-Model/data/experiment_data_54`–`610`, ≈ 5.2 GB (mostly video). Manifests: `experiments/manifest_tcds_runs*.csv`.

### 6.6 Validation (`analysis/tcds/check_runs.py` → `results/run_checks.csv`)
- **Shared needs, blind/self, blind/reference:** the needs do not depend on behaviour, so an offline replay from the logged start must reproduce them. Correlation ≥ 0.9996 in every bounded run; divergence (needs or accumulated needs > 10) matches exactly.
- **Perceiving designs:** the logged distance matches the recorded ground truth or odometry (0.0 cm median error), and the logged own call matches the recorded call in ≥ 96% of samples.
- **Result:** 557/557 logged runs pass. The 20 earlier runs (D0) were replay-checked before.

### 6.7 Measures (`analysis/tcds/gazebo.py`, `analysis/metrics.py`)
All measures are taken over the first 180 s:
- **Coordination, belief, tracking:** Pearson correlations of the 10 Hz or 30 Hz series.
- **Frequency:** zero crossings / (2 × duration).
- **Detuning:** absolute frequency differences.
- **Estimate accuracy:** correlation of each robot's logged $d_p$ (ambivalence) or $d_e$ (avoidance) with Eq. (analytic coupling) evaluated on both robots' logged own accumulated needs.
- **Call reciprocity:** fraction of the child's calling samples with a caregiver call within ±30 samples.
- **Near caregiver:** child inside the central square of half the arena width, with stuck periods excluded (an exploring child staying in one cell of a 50×50 grid for longer than mean + 2 s.d.; median 8% of a run).
- **Regime per agent:** escaped; quenched (no sign change in the last 90 s); or oscillating.
- **Statistics:** mean ± 95% t-interval over runs; s.d. where n = 3 makes the interval uninformative.

### 6.8 Offline analyses (`analysis/tcds/robustness.py`, `explore.py`, `estimator_eval.py`, `sim_agents.py`)
- **Strips:** ε step 0.02; 40 random starts. Horizons 54 model units (= 180 s) and 300 units (first 100 discarded); recorded every 10 ticks.
- **Plane:** step 0.05; 20 starts per cell; 200 units.
- **Stability:** the caregiver's accumulated need is raised by $10^{-9}$…$10^{-1}$ from the identical start (εAv = 0.1, 0.2, 0.3), over 2000 s.
- **Network-assignment ablation:** each robot uses its own networks.
- **No-partner-model (notebook) model offline:** 40 random starts, with learned vs analytic coupling.
- **Estimator accuracy in the training simulator:**
  - 10 training + 4 held-out values, 5 seeds;
  - input ablations (distance, vocal, own state);
  - closed loop over 10 seeds.

### 6.9 Inverse problem (`analysis/tcds/inverse.py`, `sim_kinematic.py`)
- **Stand-in:** a kinematic replica of the Gazebo arena and behaviours, driven by the blind/self controller, with exploration speed calibrated to 0.12 m/s.
- **Training set:** 1110 runs (37 conditions × 30 seeds: 10 identical + 20 random starts; 180 s).
- **Features:**
  - child: time near caregiver, mean and s.d. of distance, child calling, distance cycles per minute;
  - dyad: adds caregiver calling and caregiver call bouts per minute.
- **Models and test:** random forests (400 trees); classes secure (ε ≤ 0.1), avoidant, ambivalent; tested on 176 Gazebo blind/self strip runs; also Gazebo-only cross-validation by seed.

### 6.10 Reproducibility
```
python3 experiments/make_runs.py
python3 experiments/run_batch.py experiments/tcds_runs.csv --workers 6     # Gazebo, Docker, X display
python3 experiments/run_batch.py experiments/tcds_runs_e6.csv --workers 6
# analysis (MiRo image Python: /workspace/.venv/bin/python, run as the host user)
check_runs.py → gazebo.py → robustness.py → estimator_eval.py → inverse.py → paper_numbers.py → figures.py
# TeX: texlive/texlive image; papers/tcds/make_overleaf_zip.sh → draft/overleaf_import/paper_draft_ver_N.zip
```
All numbers are in `analysis/tcds/results/numbers.json`.

---

## 7. Evidence table (claim → number → source → figure)

| Claim | Number(s) | Source (`numbers.json` key or script) | Figure |
|---|---|---|---|
| R1 shared needs synchronise under avoidance (180 s, random starts) | median r 0.79 (εAv 0.1), 0.94 (0.2); 1.00 after transients | `offline/strips180/avoidant/*`, `offline/strips/avoidant/*` | Fig. 4 |
| R1 shared needs anti-synchronise under ambivalence | −0.94 (0.1), −0.98 (0.2); −1.00 after transients | `offline/strips180/ambivalent/*` | Fig. 4 |
| R1 robots agree | Gazebo r +0.94 (εAv 0.2), −0.95 (εAm 0.2) | `gazebo/coupled/random/*/0.2` | Fig. 4; Table S3 |
| R1 behavioural signature | reciprocity 1.00 (εAv 0.2); 0.29 (εAm 0.2); 0.15 (εAm 0.4) | `gazebo/coupled/random/*` | Fig. 3; Fig. S6; Table S3 |
| R1 holds over 15 min | coupled reciprocity 1.00 (Av); mean 0.28–0.30, range 0.03–0.47 (Am) | `long_runs` | Fig. S7 |
| R1 shared needs escape from random starts at moderate avoidance | from εAv 0.24; 28% at 0.3, 55% at 0.4; εAm ≥ 0.44 | `offline/strips180/*/escaped`; script check | Fig. 4; Section S2 |
| R2 coordination lost (180 s) | 0.10 (εAv 0.1), −0.04 (0.2); −0.38 (εAm 0.1), −0.08 (0.2) | `offline/strips180/*` | Fig. 4 |
| R2 lost after transients | \|median r\| ≤ 0.11 for ε 0.1–0.4 | `robust_strips.npz` (script check) | Fig. S4 |
| R2 belief persists | −0.94…−0.99 (εAm 0.1–0.4); −0.50, −0.81 (εAv 0.1, 0.2) | `offline/strips180/*/decoupled/belief` | Fig. 4 (dotted); Fig. S4 |
| R2 partner model does not follow partner | tracking ≈ 0.10 (εAv 0.1), 0.09 (εAm 0.2) | `offline/strips180/*/decoupled/track` | Fig. S4 |
| R2 detuning mechanism | 24.8 vs 31.5 mHz (εAm 0.2, offline); 6.1 mHz model–partner (Gazebo pooled) | `robust_strips` freq; `anchoring/decoupled/detune_model` | Fig. 5; Fig. 7c |
| R2 on robots | r −0.13, belief −0.95, tracking 0.19, reciprocity 0.55 (pooled εAm 0.1–0.4) | `anchoring/decoupled` | Fig. 3; Fig. 7; Table S3 |
| R2 robust to network assignment | own networks: r 0.01 (εAv 0.1), 0.00 (εAm 0.2) | `nets` | Section S3 |
| R2 plane | oscillating cells: median \|r\| 1.00 (coupled) vs 0.05 (private); 94% / 89% of starts escape | `grid` | Fig. S1 |
| R3 identical start looks synchronised | r = 1.00 (offline and Gazebo E1s/D0) | `explore_strips_gazebo`; `gazebo/decoupled/identical/avoidant/*` | Fig. 4 |
| R3 instability | 10⁻⁷ → 704 / 421 s; 10⁻³ → 322 / 188 s; 0.1 → 17–18 s (εAv 0.1 / 0.2) | `stability` | Fig. 6 |
| R3 shared needs stable | final r 1.00 for every perturbation | `stability/coupled_r_end` | Fig. 6 |
| R4 anchored partner model tracks partner | 0.76 ± 0.11 vs 0.13–0.31 | `anchoring/*/track_r` | Fig. 7b |
| R4 coordination returns | −0.62 ± 0.14 vs −0.06…−0.20; shared −0.93 | `anchoring/*/r` | Fig. 7a |
| R4 detuning removed | 0.2 mHz vs 3.5–6.1 mHz; own–partner 1.6 vs 0.5 (shared), 3.7–5.6 (other private) | `anchoring/*/detune_*` | Fig. 7c |
| R4 reciprocity | 0.35 (anchored) vs 0.25 (shared) vs 0.47–0.55 | `anchoring/*/reciprocity` | Fig. 7e |
| R4 estimate equally good in all designs (Am) | child estimate r ≈ 0.84–0.86 | `anchoring/*/est_r_c` | Fig. 7d |
| R4 avoidance | anchored at εAv 0.1: tracking 0.79, r 0.60 | `anchoring/*/avoidant0.1` | text; Fig. S3 |
| R4 cost | anchored escapes from εAm 0.6, εAv 0.2 | `gazebo/baseline-observed-gt/random/*` | text; Fig. S3; Table S2 |
| R4 estimate needs perception under avoidance | perceiving 0.97 (Av), 0.84 (Am); blind −0.51 (self), −0.80 (reference) | `estimation/*` | Section S3; Fig. S3; Table S2 |
| R4 caregiver's estimate poor | −0.48 … 0.27 | `estimation/*/est_p` | Table S2 |
| Offline estimate accuracy | RMSE 0.028/0.037 (physical), 0.041/0.014 (emotional), ε ≥ 0.1 | `estimator_offline/*/full` | Fig. S2 |
| Offline ablations | own state removed: R² 0.26/0.44; distance removed: 0.98/0.80; silent calls: 0.63/−0.12 | `estimator_offline` | Fig. S2 |
| No partner model (offline) | child estimate 0.69/0.67, carer −0.15/−0.24 | `notebook_offline` | Section S3 |
| R5 reciprocity wanders in private dyads | 0.18–1.00 over 900 s; r oscillates around 0 | `long_runs/decoupled/*` | Fig. S7 |
| R5 inverse problem | child 60% (106/176); dyad 89% (156/176); identical 88%, random 89%; Gazebo CV 56% / 74%; MAE 0.18–0.19 → 0.05–0.07 | `inverse` | Table I; Fig. S10 |
| R5 caregiver carries the information | dyad-model importance: caregiver calling 0.25 + bouts 0.26 | `inverse/dyad/importance` | Section S5 |
| Platform | 577 runs; RTF 0.97; odometry drift 1.06 m; stuck 8%; validation 557/557 | `replay`, `odometry`, `stuck_median` | Sections S4, S6 |

---

## 8. Reference map (all 37 entries in `draft/refs.bib`)

What each work says comes from its abstract or text, retrieved from OpenAlex, PubMed, Semantic Scholar, Crossref and publisher pages, and from the archive.org full text of Bowlby (1969). **V** marks claims you should verify.

### Robots, relationships and ethics
- **Prescott2021** — Prescott & Robillard, *iScience* 2021. A relational (not essentialist) view of human–robot relationships; co-operative, emotionally aligned social robots could benefit users; benefits weighed against harms. **Role:** robotics-first hook; ethics. **Where:** I-P1, VII-E.
- **Collins2013** — Collins, Millings & Prescott, AAATE 2013. Compares relationships with assistive devices to bonds with people, pets and objects; proposes a taxonomy. **Role:** robots in relationships. **Where:** I-P1, II-A.
- **Sharkey2010** — *Interaction Studies* 2010. Ethical appraisal of childcare robots: risks to children's emotional wellbeing, set against attachment disorders. **Role:** ethics of attachment-capable robots. **Where:** VII-E.
- **Robillard2018** — Robillard & Hoey, *Computer* 2018. Emotion in cognitive assistive technologies for dementia; proposes BayesACT to model behaviour and emotion. **Role:** robots already infer the partner's affective state, the estimation framing. **Where:** I-P1, II-A, VII-A.
- **Lomas2022resonance** — *Frontiers in Neurorobotics* 2022. Resonance (synchronisation, rhythmic entrainment) in human interaction and in robots/AI; a design strategy for resonant HRI. **Role:** coordination is a design goal for social robots. **Where:** I-P1, VII-A.

### Platform and robot-modelling method
- **Prescott2017miro** — HRI 2017 late-breaking report. MiRo, a pet-sized companion platform for research and therapy. **Where:** V-B. **Note:** DOI corrected (doubled prefix).
- **Mitchinson2016miro** — Living Machines 2016. MiRo's layered, brain-based control architecture (P1–P3 ≈ spinal cord, brainstem, forebrain). **Where:** V-B, III-A (layered motivation).
- **Moore2017biomimetic** — Living Machines 2017. MiRo's real-time parametric mammalian vocal synthesiser. **Role:** the physical robot's call channel used in LM; in Gazebo calls are Booleans. **Where:** V-B, limitations.
- **Collins2015saying** — Living Machines 2015. MiRo's design for companionship; emulating mammalian body language; pilot study of affective communication with light. **Role:** expressive channels (our robots' LEDs show emotional distance). **Where:** V-B (optional).
- **Mitchinson2011** — biomimetic robots as scientific models (whiskered rat robots). **Role:** why build robot models. **Where:** I-P4 or V.
- **Prescott2018** — Handbook of Living Machines (OUP 2018). Biomimetic and biohybrid systems. **Role:** context for robot models. **Where:** I-P4 (optional).

### Attachment theory and measurement
- **Bowlby1982** — *Attachment* (vol. 1, 2nd ed.). Verified in the 1969 Pelican text:
  - **control system with set-goals** (ch. "Instinctive behaviour"; thermostat analogy);
  - **"Working Models"** section in ch. 5 "Behavioural Systems Mediating Instinctive Behaviour" (≈ pp. 110–111);
  - the **"dynamic equilibrium" of four classes of behaviour** and "in a happy couple … together in harmony" (≈ p. 287; npj says ch. 13);
  - the **child's working models of mother, self, and "how each interacts"**, "only gradually … fitted for its role" (≈ p. 418, ch. "Developments in the Organization of Attachment Behaviour").
  - **Role:** the theory spine (dynamic equilibrium → coordination; working models → private copies; "fitted" → anchoring). **Where:** I-P2/P3, II-B, IV-A, VII-B.
  - **V:** page and chapter numbers in the 1982 edition.
- **Ainsworth1978** — *Patterns of Attachment*. The SSP; classification; infant SSP behaviour related to maternal behaviour at home; maternal caregiving and interaction scales. **Role:** test-case paradigm; caregiver responsiveness predicts infant pattern, cf. R5. **Where:** I-P2, II-B, VI-E. **V:** 1978 vs 2015 edition.
- **Fraley2003infant** — *Developmental Psychology* 2003. Taxometric analysis of SSP data (n = 1139): variation is largely continuous, not categorical. **Role:** continuous ε parameters. **Where:** I-P2, II-B.
- **Gagliardi2021our** — *Frontiers in Psychology* 2021. Attachment–Personality Model: seven attachment dimensions, each induced by a caregiving feature and acquired by imprinting. **Role:** dimensional theory. **Where:** II-B.
- **Gagliardi2022** — *Frontiers in Psychology* 2022. Agent-based SSP with point agents, drives and needs; parameters avoidance, ambivalence, caregiver insensitivity and unresponsiveness. Avoidance is "emotional" (insensitivity), ambivalence "physical" (unresponsiveness). **"Each agent can perceive the other's need level"**; emotional separation is inferred from the caregiver's exploration rate and perceived distance from physical separation. Qualitative fit. **Role:** the closest precedent, sharing the shortcut we remove; source of the physical/emotional coupling signals. **Where:** I-P3, II-C, III-B. **Fix:** the old draft's "infer one another's state" is inaccurate.
- **Feldman2017** — *Trends in Cognitive Sciences* 2017. Four mammalian bonds; oxytocin–dopamine crosstalk with "time sensitivity/pulsatility"; human attachments characterised by **biobehavioral synchrony**. **Role:** synchrony as the hallmark of attachment; pulsatility hints at event-based coupling. **Where:** I-P2, II-B, VII-B.
- **Chow2018representing** — *Psychometrika* 2018. Regime-switching differential-equation models fitted to mother–infant movement in the SSP, with shifts between staying close and exploring. **Role:** dynamical description of SSP proximity; route to testing predictions. **Where:** I-P2, II-B, VII-B.
- **Prince2021** — *Infant Behavior and Development* 2021. 49 one-year-olds tracked with Kinect and LENA; objective measures (contact duration, approach velocity, crying) predict 8/9 expert scales and ≈ half the variance of the attachment dimensions. **Role:** reading attachment from objective behaviour, cf. R5. **Where:** I-P2, VI-E, VII-B.

### Computational and dynamical models
- **Petters2019attachment** — *Developmental Psychology* 2019. The development of Bowlby's control-systems theory; comparison with cognitive science; prospects for computational modelling. **Where:** II-C.
- **Stevens2009dynamic** — *IEEE TAMD* 2009 (TCDS's predecessor). A homeostatic ODE model of infant attachment (opioid proximity-seeking, NE arousal) simulating the SSP; caregiver responsiveness and temperament as parameters; PCA regions for secure, anxious and avoidant. **Role:** the closest dynamic precedent in this journal's lineage; its caregiver is a parameter, not an agent with a model of the child. **Where:** II-C, VII-D.
- **Buono2006** (formerly `Atkinson2006`) — **Buono, Chau, Lewis, Madras, Pugh, Rossi & Witelski (2006)**, *Mathematical Models of Mother/Child Attachment*, Fields–MITACS Industrial Problem Solving Workshop report. Formulates models of the SSP in which the mother regulates the child's anxiety and serves as a secure base. **Where:** II-C. **V:** bib authors (LM listed "Atkinson et al."); the report link found is dead.
- **Canamero2011** — *International Journal of Humanoid Robotics* 2006. A perception–action architecture for imprinting-like attachment of a robot to a caregiver; flexible imprinting; adaptation through distress/comfort interactions. **Role:** robot attachment precedent (robot as child of a human). **Where:** II-C. **V:** year 2006 vs 2011.
- **Wilson2017** — *Royal Society Open Science* 2017. A model of rodent filial huddling emerging from thermoregulatory huddling through associative learning. **Role:** emergent mammalian social behaviour. **Where:** II-C (brief).
- **Wilson2021** — *Phil. Trans. B* 2022. Constraint closure; layered control architectures scaffolded across timescales. **Role:** multi-timescale outlook (development of attachment parameters). **Where:** VII-D.
- **Jimenez2020framework** — Living Machines 2020. A four-stage dynamical framework for motivational conflict using multi-well potentials; switching by bifurcation or stochastic escape. **Role:** origin of the motivational oscillator. **Where:** III-A.
- **Prescott2025plm** — *Psychology of Learning and Motivation* 2025 (abstract not retrievable; content from the npj draft's use). Layered brain architecture for motivation through dynamical systems, computational neuroscience and robotics. **Where:** III-A, VII-D. **V:** confirm description (own chapter).
- **Bertram2017** — *Mathematical Biosciences* 2017. Fast–slow analysis of multi-timescale systems (relaxation oscillations, bursting, canards). **Role:** analysis method. **Where:** III-A.
- **Craig2002** — *Nature Reviews Neuroscience* 2002. Interoception: an afferent system representing the physiological condition of the body, "the material me", a basis of feelings. **Role:** why needs are private. **Where:** I-P1, III-A. **Note:** keep the article; npj cited a 2015 book.

### Coordination dynamics and coupled oscillators
- **Kelso1995** — *Dynamic Patterns* (MIT Press). Coordination dynamics: relative phase as the collective variable; in-phase/anti-phase stability and transitions; symmetry breaking by frequency detuning, which shifts the relative phase and produces phase wandering and metastability. **Role:** interpretive frame for R2 (detuning → loss of locking) and R4 (anchoring removes detuning). **Where:** II-C, VI-B/D, VII-A. **V:** chapter for the detuning/symmetry-breaking treatment. Kelso, DelColle & Schöner (1990) is the classic paper, but it is not in the bib, so it goes in as a placeholder.
- **Thelen1994** — *A Dynamic Systems Approach to the Development of Cognition and Action*. Development as self-organisation; exploration and selection; stability and change. **Where:** II-C.
- **Ermentrout2019** — *Phil. Trans. A* 2019. Review of weakly coupled oscillator theory (phase reduction), extensions to non-smooth systems and isostable reduction, and the stability of phase-locked solutions. **Role:** theory of locking. **Where:** II-C, VI-B. **V:** use only general statements.
- **Dixit2019dynamics** — *Physics Letters A* 2019. Two coupled Van der Pol oscillators: attractive mean-field coupling → complete synchrony; repulsive → anti-synchronisation and suppression of oscillation; both together → stabilisation of the fixed point; inhomogeneous steady state; transitions depend on damping. **Role:** dynamical repertoire of our coupling. **Where:** III-B, VI-A.
- **Helm2012coupled** — *Emotion* 2012. Coupled linear oscillators applied to romantic partners' respiration and heart rate; cross-partner coupling detected. **Where:** II-C.
- **Palumbo2016interpersonal** — *Personality and Social Psychology Review*. Systematic review of interpersonal autonomic physiology and physiological synchrony. **Where:** II-C. **V:** year 2016 vs 2017.
- **Mirollo1990** — *SIAM Journal on Applied Mathematics* 1990. Pulse-coupled integrate-and-fire oscillators synchronise "for almost all initial conditions". **Role:** a route to anchoring via calls; contrast with R3. **Where:** VI-C, VII-A.

### This project
- **Htet2024** — LM 2024. The coupled-oscillator model, its dynamical analysis, and the physical MiRo dyad; "measures of physical and emotional distance … inferred from sensory data". **Role:** the starting point and its shortcut. **Where:** throughout; relation stated in Section I (P6) and the first-page footnote.

### Citation placeholders (`\citeneeded`) in the draft
Each needs a real source found by you:
1. user-state estimation / theory of mind in social robotics;
2. internal or forward models of other agents;
3. learning coupling functions or interaction dynamics with neural networks;
4. state estimation / observers for coupled oscillators;
5. symmetry breaking by frequency detuning in coordination dynamics (e.g., Kelso, DelColle & Schöner 1990) — new;
6. the extended analysis of the coupled model (npj draft), if posted as a preprint;
7. ROS and Gazebo;
8. AprilTag;
9. random forests;
10. ethics of affect-adaptive social robots;
11. *(optional, new)* Bowlby 1973 or Bretherton on internal working models, if a fuller IWM account is wanted.

### Bib issues to check
- Atkinson → Buono et al.: done in `refs.bib` (key `Buono2006`, comment on the entry); authors still to verify.
- Bowlby page numbers (1982 edition).
- Gagliardi claim wording.
- Craig: article vs book.
- Cañamero year (2006).
- Palumbo year.
- Ainsworth edition.
- Wilson2021 key vs 2022 year.
- MiRo DOI (fixed).
- LM's ref. 29 (Paxton, MESA astrophysics) must never be cited.

---

## 9. Writing rules for the rewrite
1. Every section opens with the question it answers and closes with the answer, in one sentence each.
2. Each results subsection is claim (topic sentence) → figure → mechanism → meaning for robots (and, where it applies, for attachment).
3. At most two numbers per sentence; prefer one decisive number per claim. Everything else goes in figures and tables.
4. No bullet lists in the introduction, results or discussion. Contributions are in prose (three or four sentences).
5. Use the vocabulary of Section 5 verbatim. Code names appear only in Appendix F.
6. Each reference does a job in the argument (Section 8). No citation lists without a sentence of what each work showed.
7. Robots first: every result states what a robot would see or do differently.
8. Honest boundaries: limitations are stated where they bear on a claim, and collected once in the discussion.
9. No text reused from LM (8-gram overlap check against the LM PDF and the npj source).

---

## 10. Open items for the authors
1. Bowlby page and chapter numbers in the 1982 edition (Section 8).
2. Atkinson2006 authorship (Buono et al.); Kelso chapter for detuning.
3. Network assignment: caregiver networks for both robots in the blind designs (original controller) vs own networks in the perceiving designs. Keep this split? (Ablation: the loss of coordination does not depend on it.)
4. The training `conv1d` filter weights the oldest sample most. Intended? (Reproduced as trained.)
5. Escape at strong (and, from random starts, moderate) avoidance: report as a model property, or add a bounded variant?
6. Whether the no-partner-model (notebook) design belongs in the main text or Appendix C (planned: appendix).
7. TCDS page limit; whether a physical-robot replication of the anchored design is planned before submission.
8. Whether to post the npj manuscript as a preprint so the extended coupled-model analysis can be cited.
9. ROS / Gazebo / MDK versions; repository URL; funding text.
10. Runs 17–33 (older 1-minute runs): which controller produced them? Not used.
