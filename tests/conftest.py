from __future__ import annotations

import pytest

import vibespatial.api as geopandas
from vibespatial import ExecutionMode
from vibespatial.testing import (
    SyntheticSpec,
    assert_matches_shapely,
    cuda_runtime_available,
    device_residency_guard,
    generate_lines,
    generate_points,
    generate_polygons,
    get_oracle_config,
    resolve_dispatch_modes,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("vibespatial")
    group.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests marked 'gpu' and include GPU dispatch parametrization when CUDA is available.",
    )
    group.addoption(
        "--dispatch-mode",
        action="append",
        choices=[mode.value for mode in ExecutionMode],
        default=[],
        help=(
            "Restrict dispatch-aware tests to one or more runtime modes. "
            "Repeatable; defaults to cpu, plus gpu when --run-gpu is set and CUDA is available."
        ),
    )


def _requested_dispatch_modes(config: pytest.Config) -> tuple[ExecutionMode, ...]:
    return tuple(ExecutionMode(value) for value in config.getoption("dispatch_mode"))


def _gpu_tests_requested(config: pytest.Config) -> bool:
    requested_modes = _requested_dispatch_modes(config)
    return config.getoption("run_gpu") or ExecutionMode.GPU in requested_modes or cuda_runtime_available()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "dispatch_mode" not in metafunc.fixturenames:
        return

    modes = resolve_dispatch_modes(
        _requested_dispatch_modes(metafunc.config),
        cuda_available=cuda_runtime_available(),
        run_gpu=_gpu_tests_requested(metafunc.config),
    )
    metafunc.parametrize("dispatch_mode", modes, ids=[mode.value for mode in modes])


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _gpu_tests_requested(config):
        if cuda_runtime_available():
            return
        reason = "CUDA Python runtime not available for gpu-marked tests"
    else:
        reason = "pass --run-gpu to include gpu-marked tests"

    skip_gpu = pytest.mark.skip(reason=reason)
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    return cuda_runtime_available()


@pytest.fixture
def dispatch_selection(dispatch_mode: ExecutionMode):
    return geopandas.get_runtime_selection(dispatch_mode)


@pytest.fixture
def auto_runtime_selection():
    return geopandas.get_runtime_selection()


@pytest.fixture
def synthetic_dataset():
    def _factory(spec: SyntheticSpec):
        generators = {
            "point": generate_points,
            "line": generate_lines,
            "polygon": generate_polygons,
        }
        return generators[spec.geometry_type](spec)

    return _factory


@pytest.fixture
def strict_device_guard():
    """Activate runtime device-residency enforcement.

    Any D2H transfer (cupy .get(), asnumpy, numpy.asarray on device data)
    inside this fixture's scope raises DeviceResidencyViolation immediately.

    Allowed callers (materialization, oracle comparison) are exempted.

    Usage::

        def test_my_gpu_op(strict_device_guard):
            result = my_gpu_operation(device_data)
            # Raises if my_gpu_operation transfers to host
    """
    with device_residency_guard("test"):
        yield


@pytest.fixture
def oracle_runner(request: pytest.FixtureRequest):
    default_config = get_oracle_config(request.node.function)

    def _runner(operation, *args, dispatch_mode: ExecutionMode = ExecutionMode.CPU, **kwargs):
        config = kwargs.pop("config", default_config)
        return assert_matches_shapely(
            operation,
            *args,
            dispatch_mode=dispatch_mode,
            config=config,
            **kwargs,
        )

    return _runner
