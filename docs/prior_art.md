# Prior art survey — learned surrogates for circuit-theoretic landscape connectivity

Phase 1 deliverable (2026-09-05). Purpose: establish what already exists, what EcoFlowBench
must be compared against, and where the novelty lies. Only references whose existence and
bibliographic details were verified during this survey are listed; each carries a DOI, arXiv
ID, or persistent URL. Items I looked for but could **not** find are listed explicitly in §7.

---

## 1. Circuit theory for landscape connectivity (the ground-truth model)

**Foundations.** McRae (2006) introduced *isolation by resistance*: gene flow across a
heterogeneous landscape is modelled as current flow in a resistor network, so that the
*resistance distance* (effective resistance) between two locations predicts genetic
differentiation better than Euclidean or least-cost distance [1]. McRae, Dickson, Keitt and
Shah (2008) generalised this to ecological connectivity: a raster of per-cell resistances is
turned into a graph, Kirchhoff/Ohm equations are solved with unit current injected at one focal
node and grounded at another, and the resulting *current density* map identifies pathways and
pinch-points that integrate over all possible random-walk routes, not just the least-cost path
[2]. Dickson et al. (2019) review ten years of applications and remaining challenges, including
computational cost at large extents [3].

**Software lineage.**
- *Circuitscape* (Python, 2008–) → *Circuitscape.jl* (Julia, v5, 2019–): Anantharaman, Hall,
  Shah and Edelman report the Julia port with a preconditioned conjugate-gradient solver using an
  algebraic multigrid (AMG) preconditioner and a CHOLMOD direct solver, and 2–4× speed-ups over
  the Python version [4]. Hall et al. (2021) describe applications enabled by the Julia rewrite
  and the resulting *Omniscape.jl* [5].
- *Omniscape* (McRae et al. 2016, TNC report) applies Circuitscape in a moving window around
  every source pixel ("coreless" / omnidirectional connectivity); *Omniscape.jl* is the
  reference implementation (Landau et al. 2021, JOSS) [6, 7]. Omniscape is the engine behind
  the global protected-area connectivity map of Brennan et al. (2022, *Science*) [8], which is
  the clearest demonstration that circuit-theoretic connectivity is applied at planetary scale
  and at meaningful cost.
- *gflow* (Leonard et al. 2017) massively parallelises pairwise Circuitscape solves on HPC
  systems to reach continental extents [9].
- *ConScape* (Van Moorter et al. 2023, Julia) computes randomised-shortest-path (RSP) based
  connectivity metrics that interpolate between least-cost and random-walk (circuit) extremes
  [10]; the RSP framework is due to Panzacchi et al. (2016) and is also exposed by the R package
  *gdistance* (van Etten 2017) [11, 12]. Least-cost modelling (Adriaensen et al. 2003) remains
  the historical baseline [13].

**Why it is expensive.** Each pairwise solve is a sparse symmetric positive-(semi)definite
linear system of size ≈ number of non-NoData pixels. Theory gives nearly-linear-time SDD
solvers (Spielman & Teng; Koutis, Miller & Peng 2010 [14]) and nearly-linear-time
approximation of *all* effective resistances via Johnson–Lindenstrauss sketches (Spielman &
Srivastava 2011 [15]), but practical Circuitscape workloads still rely on CG+AMG or CHOLMOD;
Omniscape multiplies the cost by the number of source windows. This is the computational gap a
learned surrogate targets.

**Inputs are uncertain by construction.** Resistance surfaces are expert- or data-derived
(Zeller, McGarigal & Whiteley 2012 review [16]); *ResistanceGA* (Peterman 2018) optimises them
against genetic data by running Circuitscape thousands of times inside a genetic algorithm [17];
Hanks & Hooten (2013) embed circuit theory in a Bayesian model whose likelihood evaluation
requires repeated resistance-distance computations [18]. Bowman et al. (2020) show current
density is robust to the *magnitude* of cost values as long as their *rank order* is preserved
[19]. All three points motivate a fast surrogate: inverse/optimisation loops need many forward
solves, and the input-space uncertainty means a surrogate must generalise across resistance
tables, which is why EcoFlowBench includes multiple tables and an OOD-table split.

## 2. Machine learning *for* connectivity (adjacent, not surrogate)

- Deep reinforcement learning for connectivity conservation planning (Equihua, Beckmann &
  Seppelt 2024, MEE) optimises which land to protect; it uses connectivity metrics as reward
  and would directly benefit from a fast forward model [20].
- Graph-based optimisation of connectivity (GECOT, Hamonic et al. 2025, MEE) and the RSP
  sensitivity work in ConScape are optimisation layers on top of exact solvers [21, 10].
- Deep learning in landscape genetics (e.g. *disperseNN*, Smith et al. 2023) infers dispersal
  parameters from genotypes with CNNs but does not emulate a connectivity solver [22].

None of these learn a map from resistance raster to solver output.

## 3. Learned surrogates for elliptic / Laplacian problems (the ML lineage)

Circuitscape's linear system is the 5-/9-point finite-difference discretisation of a
variable-coefficient Poisson equation ∇·(σ∇φ) = f with point sources and Dirichlet grounds,
where σ = 1/R is conductance. The relevant surrogate literature is therefore the neural-operator
and PDE-emulator literature:

- **Fourier Neural Operator** (Li et al., ICLR 2021) established Darcy flow — exactly this
  operator, with a random-field coefficient — as the canonical steady-state benchmark [23];
  the *neuraloperator* library (Kossaifi et al. 2024) is the maintained reference
  implementation we pin as a baseline [24].
- **U-Net** (Ronneberger et al. 2015) [25] and **Swin-Unet** (Cao et al., ECCVW 2022) [26]
  are the standard convolutional and transformer encoder–decoders; PDEArena (Gupta &
  Brandstetter 2022) found modern U-Nets to be very strong PDE surrogates [27].
- **Graph networks** (Battaglia et al. 2018 [28]; MeshGraphNets, Pfaff et al. ICLR 2021 [29])
  operate on the discretisation graph itself, which for Circuitscape is literally the resistor
  network — the natural inductive bias for exact edge conductances and NoData holes.
- **Learned preconditioners / neural solvers** for Poisson systems (Lan et al. 2024, ICML,
  neural-preconditioned solver for mixed BCs [30]; "Learning preconditioners for conjugate
  gradient PDE solvers", ICML 2023 [31]) are the complementary route: accelerate, rather than
  replace, the solver. EcoFlowBench's stored ground truth and solve-time statistics support
  evaluating that route too.
- Effective resistance is also an object of interest in graph ML as a **positional encoding**
  (e.g. Wang et al. ICLR 2022 [32]); a fast learned approximation of Reff on grid graphs is of
  independent interest.

## 4. Benchmark datasets we model ourselves on

| Dataset | Domain | Design features EcoFlowBench adopts |
|---|---|---|
| **PGLearn** (Klamkin, Tanneau, Van Hentenryck 2025, arXiv:2505.22825) [33] | AC/DC optimal power flow | fixed reference solver; standardised instance families; multiple formulations per instance; complete primal/dual solution data; generation code released as a package; official splits; baseline table; HF hosting |
| OPFData (Lovett et al. 2024, arXiv:2406.07234) [34] | AC-OPF with topological perturbations | large scale, structural (graph) perturbations, HF-style distribution |
| OPF-Learn (Joswig-Jones et al. 2021, arXiv:2111.01228) [35] | AC-OPF | representative sampling of the feasible input space |
| **PDEBench** (Takamoto et al., NeurIPS 2022 D&B) [36] | 1–3D PDEs incl. Darcy | HDF5 shards, per-PDE configs, FNO/U-Net/PINN baselines, forward + inverse tasks |
| PDEArena (Gupta & Brandstetter 2022) [27] | NS, shallow water, Maxwell | many models, one codebase, strong U-Net baselines |
| **The Well** (Ohana et al., NeurIPS 2024 D&B) [37] | 16 physics simulations, 15 TB | uniform HDF5 layout + metadata, per-dataset cards, streaming loaders |
| AirfRANS (Bonnet et al., NeurIPS 2022 D&B) [38] | RANS over airfoils | explicit OOD tasks (Reynolds / angle-of-attack extrapolation), scarce-data regime |
| ConDiff (2024, arXiv:2406.04709) [39] | diffusion with high-contrast coefficients | *contrast* as a difficulty axis — directly analogous to our resistance dynamic-range ladder |

Documentation standards: *Datasheets for Datasets* (Gebru et al. 2021) [40]; the NeurIPS
2026 Evaluations & Datasets track requires hosting on Hugging Face / Dataverse / Kaggle /
OpenML, mandatory **Croissant** metadata (core + RAI fields), and long-term availability [41].
(2026 deadlines have passed — full paper May 6, 2026 — so the realistic target is the 2027
cycle or another venue; flagged in the phase report.)

## 5. Synthetic landscape generation

Neutral landscape models are the standard way to create controllable synthetic landscapes:
*NLMpy* (Etherington, Holland & O'Sullivan 2015, MEE) implements random clusters, planar and
distance gradients, midpoint-displacement fractals and mosaics in NumPy [42]. EcoFlowBench
uses it directly (pinned) plus Gaussian random fields and barrier overlays.

## 6. Novelty assessment

Verified gaps (as of 2026-09-05; see §7 for the searches performed):

1. **No public dataset of solver-computed circuit-theory connectivity outputs exists** in the
   PGLearn / PDEBench sense (standardised instances, official splits, OOD test sets, baselines,
   generation code). Individual studies release their own current maps, but not as an ML
   benchmark.
2. **No published learned surrogate of Circuitscape / Omniscape** was found. The closest
   items are (a) Darcy-flow neural operators (same PDE, but smooth log-normal coefficients,
   no point sources, no NoData, no 8-neighbour graph semantics, no effective-resistance target)
   and (b) generic Poisson neural preconditioners.
3. Distinctive elements EcoFlowBench adds beyond Darcy-style benchmarks: point-source /
   grounded configurations and pairwise cumulative maps (T1), an effective-resistance matrix
   target (T2), source-strength/ground rasters (T3), the windowed Omniscape operator (T4),
   real-world covariate stacks with multiple resistance tables, NoData masks, extreme
   coefficient contrast (10–10⁴), and explicit OOD splits by region, scale, table, contrast and
   synthetic→real.

## 7. Searches performed and negative results

Web searches (Sept 2026) on: "Circuitscape" + {neural network, deep learning, surrogate,
emulator, U-Net, graph neural network, neural operator}; "Omniscape" + {machine learning,
emulator}; "landscape connectivity" + {surrogate, emulator, neural operator, current density
prediction}; "effective resistance" + {neural network prediction, learned}; "connectivity
benchmark dataset machine learning". None returned a paper or dataset that learns
resistance-raster → current-map / Reff mappings. Searches on arXiv listings for 2025–2026 with
the same terms were also negative. This is a search-based finding, not a proof of absence; the
phase report recommends a final check on Google Scholar / Semantic Scholar by the owner before
submission.

## References

1. McRae, B. H. (2006). Isolation by resistance. *Evolution* 60(8):1551–1561. doi:10.1111/j.0014-3820.2006.tb00500.x
2. McRae, B. H., Dickson, B. G., Keitt, T. H., & Shah, V. B. (2008). Using circuit theory to model connectivity in ecology, evolution, and conservation. *Ecology* 89(10):2712–2724. doi:10.1890/07-1861.1
3. Dickson, B. G., et al. (2019). Circuit-theory applications to connectivity science and conservation. *Conservation Biology* 33(2):239–249. doi:10.1111/cobi.13230
4. Anantharaman, R., Hall, K., Shah, V. B., & Edelman, A. (2020). Circuitscape in Julia: High performance connectivity modelling to support conservation decisions. *Proceedings of the JuliaCon Conferences* 1(1):58. doi:10.21105/jcon.00058 (arXiv:1906.03542)
5. Hall, K. R., Anantharaman, R., Landau, V. A., Clark, M., Dickson, B. G., Jones, A., et al. (2021). Circuitscape in Julia: Empowering dynamic approaches to connectivity assessment. *Land* 10(3):301. doi:10.3390/land10030301
6. McRae, B. H., Popper, K., Jones, A., Schindel, M., Buttrick, S., Hall, K., Unnasch, B., & Platt, J. (2016). Conserving Nature's Stage: Mapping omnidirectional connectivity for resilient terrestrial landscapes in the Pacific Northwest. The Nature Conservancy, Portland, OR. (Technical report; cited via [7].)
7. Landau, V. A., Shah, V. B., Anantharaman, R., & Hall, K. R. (2021). Omniscape.jl: Software to compute omnidirectional landscape connectivity. *Journal of Open Source Software* 6(57):2829. doi:10.21105/joss.02829
8. Brennan, A., Naidoo, R., Greenstreet, L., Mehrabi, Z., Ramankutty, N., & Kremen, C. (2022). Functional connectivity of the world's protected areas. *Science* 376(6597):1101–1104. doi:10.1126/science.abl8974
9. Leonard, P. B., Duffy, E. B., Baldwin, R. F., McRae, B. H., Shah, V. B., & Mohapatra, T. K. (2017). gflow: software for modelling circuit theory-based connectivity at any scale. *Methods in Ecology and Evolution* 8(4):519–526. doi:10.1111/2041-210X.12689
10. Van Moorter, B., Kivimäki, I., Panzacchi, M., Saerens, M., et al. (2023). Accelerating advances in landscape connectivity modelling with the ConScape library. *Methods in Ecology and Evolution* 14(1):133–145. doi:10.1111/2041-210X.13850
11. Panzacchi, M., Van Moorter, B., Strand, O., Saerens, M., Kivimäki, I., St. Clair, C. C., Herfindal, I., & Boitani, L. (2016). Predicting the continuum between corridors and barriers to animal movements using Step Selection Functions and Randomized Shortest Paths. *Journal of Animal Ecology* 85(1):32–42. doi:10.1111/1365-2656.12386
12. van Etten, J. (2017). R package gdistance: Distances and routes on geographical grids. *Journal of Statistical Software* 76(13). doi:10.18637/jss.v076.i13
13. Adriaensen, F., Chardon, J. P., De Blust, G., Swinnen, E., Villalba, S., Gulinck, H., & Matthysen, E. (2003). The application of 'least-cost' modelling as a functional landscape model. *Landscape and Urban Planning* 64(4):233–247. doi:10.1016/S0169-2046(02)00242-6
14. Koutis, I., Miller, G. L., & Peng, R. (2010). Approaching optimality for solving SDD linear systems. *FOCS 2010*, 235–244. doi:10.1109/FOCS.2010.29 (arXiv:1003.2958)
15. Spielman, D. A., & Srivastava, N. (2011). Graph sparsification by effective resistances. *SIAM Journal on Computing* 40(6):1913–1926. doi:10.1137/080734029 (STOC 2008; arXiv:0803.0929)
16. Zeller, K. A., McGarigal, K., & Whiteley, A. R. (2012). Estimating landscape resistance to movement: a review. *Landscape Ecology* 27(6):777–797. doi:10.1007/s10980-012-9737-0
17. Peterman, W. E. (2018). ResistanceGA: An R package for the optimization of resistance surfaces using genetic algorithms. *Methods in Ecology and Evolution* 9(6):1638–1647. doi:10.1111/2041-210X.12984
18. Hanks, E. M., & Hooten, M. B. (2013). Circuit theory and model-based inference for landscape connectivity. *Journal of the American Statistical Association* 108(501):22–33. doi:10.1080/01621459.2012.724647
19. Bowman, J., Adey, E., Angoh, S. Y. J., Baici, J. E., Brown, M. G. C., Cordes, C., Dupuis, A. E., Newar, S. L., Scott, L. M., & Solmundson, K. (2020). Effects of cost surface uncertainty on current density estimates from circuit theory. *PeerJ* 8:e9617. doi:10.7717/peerj.9617
20. Equihua, J., Beckmann, M., & Seppelt, R. (2024). Connectivity conservation planning through deep reinforcement learning. *Methods in Ecology and Evolution* 15(4):779–790. doi:10.1111/2041-210X.14300
21. Hamonic, F., et al. (2025). GECOT: Graph-based ecological connectivity optimization tool. *Methods in Ecology and Evolution*. doi:10.1111/2041-210X.70055
22. Smith, C. C. R., Tittes, S., Ralph, P. L., & Kern, A. D. (2023). Dispersal inference from population genetic variation using a convolutional neural network. *Genetics* 224(2):iyad068. (bioRxiv 10.1101/2022.08.25.505329; PMC10213498)
23. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. *ICLR 2021*. arXiv:2010.08895
24. Kossaifi, J., Kovachki, N., Li, Z., Pitt, D., Liu-Schiaffini, M., George, R. J., Bonev, B., Azizzadenesheli, K., Berner, J., Duruisseaux, V., & Anandkumar, A. (2024). A library for learning neural operators. arXiv:2412.10354
25. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*, LNCS 9351:234–241. doi:10.1007/978-3-319-24574-4_28
26. Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q., & Wang, M. (2022). Swin-Unet: Unet-like pure transformer for medical image segmentation. *ECCV 2022 Workshops*, LNCS 13803. doi:10.1007/978-3-031-25066-8_9
27. Gupta, J. K., & Brandstetter, J. (2022). Towards multi-spatiotemporal-scale generalized PDE modeling. arXiv:2209.15616 (PDEArena, https://github.com/pdearena/pdearena)
28. Battaglia, P. W., et al. (2018). Relational inductive biases, deep learning, and graph networks. arXiv:1806.01261
29. Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. W. (2021). Learning mesh-based simulation with graph networks. *ICLR 2021*. arXiv:2010.03409
30. Lan, K., Gershenson, S., Kim, S., & Teran, J. (2024). A neural-preconditioned Poisson solver for mixed Dirichlet and Neumann boundary conditions. *ICML 2024*, PMLR 235. arXiv:2310.00177
31. Li, Y., Chen, P. Y., Du, T., & Matusik, W. (2023). Learning preconditioners for conjugate gradient PDE solvers. *ICML 2023*, PMLR 202.
32. Wang, H., Yin, H., Zhang, M., & Li, P. (2022). Equivariant and stable positional encoding for more powerful graph neural networks. *ICLR 2022*. arXiv:2203.00199
33. Klamkin, M., Tanneau, M., & Van Hentenryck, P. (2025). PGLearn — An open-source learning toolkit for optimal power flow. arXiv:2505.22825
34. Lovett, S., et al. (2024). OPFData: Large-scale datasets for AC optimal power flow with topological perturbations. arXiv:2406.07234
35. Joswig-Jones, T., Baker, K., & Zamzam, A. S. (2021). OPF-Learn: An open-source framework for creating representative AC optimal power flow datasets. arXiv:2111.01228
36. Takamoto, M., et al. (2022). PDEBench: An extensive benchmark for scientific machine learning. *NeurIPS 2022 Datasets and Benchmarks Track*. arXiv:2210.07182
37. Ohana, R., McCabe, M., et al. (2024). The Well: a large-scale collection of diverse physics simulations for machine learning. *NeurIPS 2024 Datasets and Benchmarks Track*. arXiv:2412.00568
38. Bonnet, F., Mazari, J. A., Cinnella, P., & Gallinari, P. (2022). AirfRANS: High fidelity computational fluid dynamics dataset for approximating Reynolds-averaged Navier–Stokes solutions. *NeurIPS 2022 Datasets and Benchmarks Track*. arXiv:2212.07564
39. ConDiff: A challenging dataset for neural solvers of partial differential equations (2024). arXiv:2406.04709
40. Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM* 64(12):86–92. doi:10.1145/3458723 (arXiv:1803.09010)
41. NeurIPS 2026 Evaluations & Datasets Track, Call for Papers. https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets (accessed 2026-09-05)
42. Etherington, T. R., Holland, E. P., & O'Sullivan, D. (2015). NLMpy: a Python software package for the creation of neutral landscape models within a general numerical framework. *Methods in Ecology and Evolution* 6(2):164–168. doi:10.1111/2041-210X.12308

### Verification notes
- Journal volume/page details for [3], [9], [10], [16], [17] were taken from publisher landing
  pages or indexing services surfaced by the searches; [6] is a grey-literature report cited
  through the Omniscape.jl JOSS paper and was not independently retrieved.
- [21] was found via the Wiley landing page (2025, MEE); the full author list beyond the first
  author was not confirmed.
- [22]'s *Genetics* citation was not independently confirmed beyond the bioRxiv/PMC records; treat the journal details as provisional.
- [31] author list is from the ICML proceedings listing; verify before citing in the paper.
- [39] author list was not retrieved; cite by arXiv ID only until confirmed.
