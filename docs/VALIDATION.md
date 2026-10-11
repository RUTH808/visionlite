# Validation performed

- Python CLI executed on the complete named public dataset.
- Reported metrics come from the saved executed run, not estimates.
- Unit tests passed locally under Python 3.12; test definitions are in tests/.
- Browser report opened and row filtering exercised on desktop and mobile widths.
- AI projects additionally passed live example inference, health endpoint, and invalid-request checks.
- GitHub Actions runs unit tests on pushes and pull requests. Current run status: https://github.com/RUTH808/visionlite/actions . Full dataset benchmarks were executed locally; CI runs the tests defined in the workflow.
- Dockerfiles and Power BI authoring templates are provided but have not been executed in Docker/Power BI Desktop here.

The recorded environment, splits and conditions are in results/metrics.json. Do not describe unrun packaging paths as validated deployments.
