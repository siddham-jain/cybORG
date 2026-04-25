"""FastAPI application for the CybOrg environment.

Wires :class:`CybOrgEnvironment` (the world simulator + reward rubrics) to the
OpenEnv HTTP/WebSocket server. Judges and trainers can hit ``/reset`` and
``/step`` directly, or use the included :class:`cyborg_env.client.CybOrg`
client for a friendlier API.
"""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install with `uv sync`."
    ) from e

try:
    from ..models import CybOrgAction, CybOrgObservation
    from .cyborg_env_environment import CybOrgEnvironment
except ModuleNotFoundError:  # pragma: no cover - container path
    from models import CybOrgAction, CybOrgObservation
    from server.cyborg_env_environment import CybOrgEnvironment


app = create_app(
    CybOrgEnvironment,
    CybOrgAction,
    CybOrgObservation,
    env_name="cyborg",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Direct-execution entry point.

    Usage:
        ``uv run --project . server``
        ``python -m cyborg_env.server.app``
    """

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
