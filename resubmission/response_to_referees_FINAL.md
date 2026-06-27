# Response to Referees — Manuscript XQ11175

*"How Focused Are LLMs? A Quantitative Study via Repetitive Deterministic Prediction Tasks"*

We thank the Editors and both Referees for their careful and constructive reading. We have
revised the manuscript to address every point raised. The most significant changes are:

1. **Statistics.** Every accuracy-versus-length curve is now shown with explicit sample sizes
   and 95% confidence intervals. Each plotted point aggregates up to *n* = 100 independent
   problem instances (10 repetitions of a batch of 10); a subset of grok-4 runs use fewer
   repetitions (*n* ≈ 20–30), now stated. Error bars are 95% confidence intervals (bootstrap over
   the repeated batches for the geometric-mean SAR; Wilson score intervals where SAR is reported as
   the instance-level success rate, i.e. integer addition and the scaling collapse).
2. **Positioning of the statistical model.** The Sherrington–Kirkpatrick (SK) mapping is now
   presented explicitly as an *effective, phenomenological* model that reproduces the observed
   crossover and yields a compact two-parameter diagnostic, rather than as a microscopic theory
   of the network. Overclaims about a first-principles link to attention have been removed.
3. **Reconciliation of the two functional forms** flagged as inconsistent (the perturbative
   expansion and the double-exponential): we now show the latter is the exponentiation of the
   leading-order correction of the former, and correct a coefficient error in the large-field
   reduction.
4. **A scaling-collapse figure** has been added, directly testing (and confirming) the universal
   crossover that the spin-glass picture predicts.

A point-by-point response follows. Line/figure references are to the revised manuscript.

---

## Referee 1

**General remarks.** We agree the manuscript sits at the physics/machine-learning interface and
is not pitched as a field-defining machine-learning advance; this is why we welcome the transfer
to Physical Review Research. We have substantially strengthened the data presentation (stated
sample sizes, error bars) and the justification and positioning of the statistical model, as
detailed below.

**1. The statement that success is "factorizable" (a product of per-step accuracies) is
imprecise; integer addition has carries.**
We agree and have rewritten the passage. The strict factorization holds only for tasks whose
output positions are computed independently (cyclic letter replacement, and Pauli-string
multiplication site-by-site). Integer addition involves carry propagation, a long-range
dependence, so the factorized picture is at best a leading-order approximation there. The
revised text states this explicitly and uses the range of dependency structures (site-local
versus carry-coupled) as part of the motivation.

**2. State the number of trials per length and show error bars.**
Done, using the already-collected data (no new runs were required). We had recorded 10
repetitions per sequence length, each scoring a batch of 10 problems, i.e. up to *n* = 100
independent instances per point. The revised figures show 95% confidence intervals on every
point; the sample size is stated in each caption and in the Methods. The point-to-point scatter
previously noted is consistent with binomial sampling at this *n* and is now bounded by the
displayed intervals.

**3. The SK model is poorly motivated: no Z₂ symmetry; why is an incorrect configuration ever
lower-energy; where is the frustration?**
We have clarified the physical picture and now present the SK form as an effective model chosen
for the minimal ingredients that produce a crossover, not derived from a microscopic symmetry.
Specifically: (i) the external field *h* explicitly breaks the *s* → −*s* symmetry, so there is
no Z₂ degeneracy between the all-correct and all-incorrect states; (ii) at small *N* the field
selects the all-correct configuration (paramagnetic regime), whereas at large *N* the disorder
energy (∝ *N*²) overtakes the field, so the all-correct configuration is no longer favored, and
in the strong-disorder/weak-field regime an incorrect configuration can become energetically
competitive; (iii) the frustration is the inability to simultaneously satisfy the competing
random-sign couplings, which compete with the ordering field. This material now appears in the
Hypothesis section.

**4. A simpler model — uniform ferromagnetic couplings with a random field — instead of
disordered couplings: was it excluded?**
Yes, and we now make the argument explicit. A model with a field alone (uniform, or a
site-dependent random field) but no pairwise coupling factorizes over sites, giving log SAR
strictly linear in *N* — a single exponential with no crossover. Uniform ferromagnetic couplings
would instead drive a smooth mean-field transition, not the sharp, sample-to-sample crossover
observed. The disorder (random-sign coupling) is what produces the super-linear-in-*N* term and
hence the cliff.

**5. The divide-and-conquer "validation": (a) it follows from the empirical law, not the
SK-specific equations; (b) it works for the Pro model but not Flash; (c) it should be tested on
the other tasks.**
(a) Agreed; we have rewritten the claim so that the prediction follows from the *empirical*
accumulation law and does not rely on the SK-specific equations — it therefore tests the
accumulation law itself. (b) The Pro/Flash difference is consistent with the per-call overhead:
a weaker model loses more to the bookkeeping of decomposition than it gains from shorter
sub-problems. We now state this and no longer present the Flash curve as a success. (c) We agree
that extending the divide-and-conquer test to the other task families is valuable and flag it
explicitly as future work; Pauli multiplication is the natural first testbed because its
sub-problems are exactly site-local and recombination is unambiguous.

**6. The closed-form expansion does not give the double-exponential form; reconcile.**
We thank the Referee for catching that the two forms were presented without connecting them. We
have made the relationship explicit and honest in Appendix A. The double-exponential is obtained
by exponentiating the leading-order (O(J₀²)) correction of the perturbative result; this yields
the empirical law with β₀ = e^(−2h) and α ≈ e^(2J₀²). On re-checking the algebra we found and
corrected an error in the large-field reduction (the O(J₀⁴) coefficient is −4 J₀⁴(N−1)², not
+2 J₀⁴(N−1)²); since a strict exponentiation would require +2, we now state plainly that the
double-exponential is a phenomenological closed form valid in the small-J₀, large-*h* regime,
**not** a term-by-term resummation. The physically essential conclusion — a positive,
super-linear *N*² contribution to −log SAR that drives the cliff — follows already from the
leading O(J₀²) term and is unaffected.

---

## Referee 2

**General remarks.** We thank the Referee for the supportive recommendation and the precise
technical critique. We have (i) added full statistics, (ii) repositioned the SK mapping as an
effective model, (iii) confirmed the absence of tool use, (iv) specified the decoding settings,
(v) added a model-comparison test, (vi) reconciled the two functional forms, and (vii) added the
requested scaling-collapse analysis.

**1. The couplings J_ij are postulated i.i.d. Gaussian but not connected to any measurable
network quantity; the "attention-induced interference" language is vague.**
We agree and have softened the suggestive language, now presenting the couplings as an effective
form inspired by — but not derived from — the all-to-all structure of attention; they are not
claimed to equal any specific network weights. We list the microscopic connection the Referee
suggests — extracting effective J_ij from attention patterns, and testing whether perturbing
attention sparsity shifts α as predicted — as future work, since it is beyond the present scope.

**2. A real spin-glass crossover should show a scaling collapse versus N/N\*, and the N²-versus-N
energetic argument should be checked.**
We have added a scaling-collapse figure (new figure in the Analysis section): plotting SAR
against the rescaled length N/N\* (with N\* the fitted SAR = 0.5 scale) for all model–task pairs.
The curves collapse onto a single crossover curve passing through (N/N\*, SAR) ≈ (1, 0.5),
directly supporting the universal finite-size crossover predicted by the spin-glass picture. The
N²-versus-N energetic argument that drives the cliff is stated in the Hypothesis section and is
consistent with this collapse. Both use only already-collected data.

**3. No statement of how many instances per N; SAR plots show single points without error bars.**
Addressed as in Referee 1, point 2: up to *n* = 100 instances per point, 95% confidence intervals on
every point, sample size stated in captions and Methods.

**4. Confirm the models are not silently using internal tools, especially on large-N addition.**
Confirmed. All queries were issued through plain text completion/chat calls with no tools,
function-calling, or code-interpreter interfaces enabled; the models cannot invoke external
calculators or solvers, and all results reflect in-context computation only. This is now stated
explicitly in the Experiment section, and the API wrapper code is released.

**5. The two-parameter fit is flexible; some fits look biased. Add a model comparison against the
α = 1 single exponential and against alternative forms.**
We added a model comparison against the single-exponential baseline (α = 1) via the Akaike
information criterion across all model–task pairs. The double-exponential is favored in all but
the highest-accuracy cases (for example, gemini-2.5-pro on integer addition, whose accuracy
decays only gradually and sits in the α → 1 limit). This shows the two-parameter form contains
the single exponential as a genuine special case and is selected by the data only where
warranted. The manuscript states this result; the released analysis code contains the per-curve
comparison.

**6. In Appendix A, the closed form does not give the double-exponential; reconcile.**
This is addressed jointly with Referee 1, point 6. We corrected the O(J₀⁴) coefficient and now
present the double-exponential as the exponentiation of the leading-order correction (a
phenomenological closed form valid in the small-J₀, large-*h* regime), not a strict resummation;
the leading-order mapping between (α, β₀) and (J₀, h) is retained.

---

We believe these revisions address the Referees' concerns and are well matched to the criteria
of Physical Review Research. We thank the Editors and Referees again for their time.
