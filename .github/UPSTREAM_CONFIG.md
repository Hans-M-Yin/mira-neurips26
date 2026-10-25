# Upstream VERL configuration

The upstream VERL CI definitions are preserved in `upstream-workflows/` for reference.
They target the upstream project's runners, registries, and deployment services.
They are stored outside `.github/workflows/`, so publishing this code does not start
those infrastructure jobs. Configure your own runners and `CI_RUNNER_ENDPOINT`
before adapting a workflow for this repository.
