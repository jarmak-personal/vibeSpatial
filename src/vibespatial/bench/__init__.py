"""vibeSpatial benchmarking framework.

Public API
----------
- ``vibespatial.bench.schema`` — ``BenchmarkResult``, ``SuiteResult``, ``TimingSummary``
- ``vibespatial.bench.catalog`` — ``@benchmark_operation``, ``list_operations``
- ``vibespatial.bench.runner``  — ``run_operation``, ``run_pipeline``, ``run_suite``
- ``vibespatial.bench.output``  — ``render_result``, ``render_suite``, ``render_list``
- ``vibespatial.bench.compare`` — ``compare``
- ``vibespatial.bench.shootout`` — ``run_shootout``, ``ShootoutResult``

CLI entry point: ``vsbench`` (see ``vibespatial.bench.cli``).
"""
