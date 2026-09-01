# capped-pricing-deterministic-work

Reproducibility package for **"Deterministic-Work Accounting for Column
Generation with Time-Capped Pricing"** — a study of how time limits on MIP
pricing oracles corrupt wall-clock comparisons in column generation, a
deterministic-work protocol to replace them, and the measurement of observable
loop signals (admission–difficulty coupling, stratified column usefulness)
on the VRPTW and on set covering by independent sets (graph coloring).

## What is included

| Directory | Content |
|---|---|
| `src/plan2_route_pool/` | VRPTW column-generation engine: covering master (CPLEX LP/IP), ESPPRC MILP pricing with solution-pool harvesting, admission strategies, threshold-rule controller, instrumented study loop |
| `src/plan2_gc/` | Graph-coloring CG: independent-set master (set covering) and MWIS MIP pricing, same instrumentation |
| `experiments/configs/` | All study configurations (YAML) reproducing every experiment of the paper |
| `data/raw/vrptw/` | Solomon 100-customer and Homberger 200-customer benchmark instances (input data only) |
| `data/vrptw_dimacs_bks.csv` | Best-known-solution provenance for bound comparisons |
| `analysis/` | Verification and analysis scripts (no plotting): independent number verification (72 claims), uncertified-termination scan, certificate calls, rule evaluation, threshold fitting, four-configuration reproducibility reruns |
| `results/raw/` | 603 raw run logs (JSON): per-iteration pricing status, wall-clock, deterministic ticks, nodes, candidate counts, admitted-column signatures, final LP bases |
| `analysis/generated_tables/` | Derived tables read by the verification script: rule evaluation, anytime certificates, certificate-call records |

Not included: manuscript sources, figures, and plotting scripts.

## Environment

- Python 3.10, IBM ILOG CPLEX 22.1.1 (with `cplex` Python API on `PATH`/`PYTHONPATH`)
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyyaml`
- Hardware reference: Apple Silicon (arm64), 4 threads per run; solver settings
  in the paper (CPLEX defaults, opportunistic parallel mode, no random seeds
  in our code)

## Reproducing the paper's experiments

```bash
# diagnostic ladder (F1–F5), e.g. for one instance:
PYTHONPATH=src python -m plan2_route_pool.cli run-study \
    --config experiments/configs/cplex_ladder_C101.yaml

# measurement corpus (R1–R3), per family:
PYTHONPATH=src python -m plan2_route_pool.cli run-study --config experiments/configs/m1_c.yaml

# graph-coloring replication:
PYTHONPATH=src python -m plan2_gc.gc_study experiments/configs/gc_m1.yaml
```

Outputs land in `results/raw/<study>/` as per-run JSON with full
per-iteration instrumentation.

## Verifying the paper's numbers

Every quantitative claim in the manuscript is re-derived from the released
logs by:

```bash
python analysis/verify_paper_numbers.py
```

which prints a PASS/FAIL line per claim and ends with `TOTAL FAILS: 0`
for the camera-ready manuscript.

Other analyses:

```bash
python analysis/false_termination_scan.py   # F2 prevalence (machine-checkable predicate)
python analysis/evaluate_rules.py           # threshold-rule vs fixed regimes
python analysis/anytime_certificates.py R1  # certificate calls + bound intervals (per family)
```

Note: these scripts expect the directory layout of the private research
workspace (`results/`, `paper/ejor_submission/generated/`); adjust the
`PLAN2` path constant at the top of each script, or re-run the studies above
to regenerate the logs locally.

## License

MIT. Benchmark instance data (Solomon, Homberger) is redistributed for
reproducibility; original sources retain their own terms.
