# Contribution-and-scope contract

- Core contribution: a corpus-conditioned lexical realization signature recurs across interfaces and generators, remains continuously readable after generator holdout, transports beyond complete fitted-pipeline nulls in two observer models, and yields a selective held-out free-generation effect in Llama 3.1 8B after strength selection on a disjoint development set.
- Contribution hierarchy: (1) selective lexical realization in the held-out Llama experiment; (2) cross-interface surface recurrence; (3) cross-generator continuous readout; (4) fitted-null lexical transport in Llama and Qwen; (5) same-token and construction diagnostics; (6) syntactic and low-dose results as scope/sensitivity evidence.
- Claim ceiling: evidence for selective control of one corpus-conditioned lexical coordinate in Llama 3.1 8B with one development-selected, single-layer 10% residual-norm controller. This is not a universal authorship factor, a guarantee across models or linguistic components, or evidence of semantic equivalence.
- Load-bearing cautions: the formal positive generation test covers Llama lexical steering only; the 20-prompt development set selected the 10% strength; prompts 34--199 and three seeds form the untouched test; formal-test semantic guardrails have not been computed; syntax remains inside the fitted-pipeline null; discourse remains an off-target axis; the earlier 5% multi-condition audit is retained as sensitivity evidence.
- Edit authority: replace title, abstract, introduction, generation method, results, discussion, conclusion, relevant captions/tables, and the generation appendix; relocate rather than delete genuine boundary evidence.

# Candidate register

| Location | Problem | Action | Evidence retained |
|---|---|---|---|
| Title/abstract | Old null result is the headline | Reframe | All formal positive effect sizes and scope |
| Introduction | Ends by denying control | Replace | Four-stage evidence chain and claim ceiling |
| Generation method | Says 5% strength was frozen for the main test | Correct | Disjoint 20-prompt development selection; 166-prompt held-out test |
| Main generation table/figure | Displays only the earlier 5% audit | Replace | New target, off-target, selectivity, CIs, seed means |
| Discussion/conclusion | Organized around failure | Replace | Positive lexical chain; syntax and generality as boundaries |
| Appendix generation audit | Treats low-dose audit as final result | Reframe and extend | Original four-condition values remain visible as low-dose sensitivity |
| Limitations | Does not name the positive test scope | Tighten | Model, axis, layer, strength, and guardrail limits |

# Live queries and deliberate non-edits

- Do not claim semantic preservation for the held-out 10% test until its formal guardrails are computed.
- Do not extend the selective-generation claim to Qwen or syntax.
- Preserve the fitted-null syntactic result and the low-dose four-condition audit in the appendix.
- Preserve all numeric results and citations outside changes directly supported by frozen artifacts.

# Revision outcome

- Reframed title, abstract, introduction, generation method, experiments, results, discussion, conclusion, limitations, generation appendix, and compute ledger around the positive lexical chain.
- Replaced the main generation table with the held-out Llama lexical target, off-target, aggregate-selectivity, and pairwise-selectivity estimates.
- Replaced the main and appendix generation figures from the frozen prompt-level CSVs; the new deterministic renderer is `scripts/plot_positive_generation_result.py`.
- Updated the accepted overview raster only at its final outcome labels: `selective lexical control` and `selective lexical effect`.
- Retained the original 5% four-condition results as a labeled low-dose sensitivity table; retained syntactic fitted-null results as a component contrast.
- Added the 0.047 GPU-process-hour / 127,476-token held-out run to the compute record; the recorded total is now approximately 3.392 GPU-process hours.

# Claim/evidence regression

- Formal-test population: prompt indices 34--199, 166 prompts, three paired seeds; matches the frozen manifest and CSV rows.
- Development/test separation: prompt indices 0--19 select the 10% endpoint; no formal result is estimated from these prompts.
- Primary target: 0.269 [0.127, 0.412], paired d 0.280, p 0.0006; matches `seed_robustness_summary.csv`.
- Primary selectivity: 0.287 [0.117, 0.459], paired d 0.255, p 0.0012; matches `seed_robustness_summary.csv`.
- Off-target and pairwise values match `group_response_summary.csv` and `pairwise_selectivity_summary.csv`.
- Claim ceilings retained in the introduction, discussion, and limitations; no formal-test semantic-preservation claim was introduced.
- Final LaTeX build: 15 pages; main text ends with the complete conclusion on page 8, references start on page 9, appendix starts on page 11.
- Compile regression: no undefined references, missing citations, multiply-defined labels, or overfull boxes; only ordinary underfull-box diagnostics remain.
- Visual regression: the main result legend no longer overlaps its title; the forced appendix column break was removed, eliminating the mostly blank terminal appendix page.
