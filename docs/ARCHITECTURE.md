# Execution and data boundaries

Inputs are downloaded to data/ (ignored by Git). A deterministic Python CLI validates and transforms them. Computed tables or trained/indexed models live in generated folders. Aggregate metrics and self-contained reports are retained under results/. Tests use small controlled inputs and cover logical edge cases. Optional local HTTP inference is available in the three AI projects. No cloud account, paid API, or hidden remote service is required.

Data sources and model libraries are versioned in requirements.txt, README source links and metrics.json manifests. Rebuild commands are documented in README. Only measured results from executed runs should be used in a resume. Distinguish model quality from pipeline unit-test success and offline results from production impact.
