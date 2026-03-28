"""FastAPI app for the Hospital Queue Management Environment."""

from openenv.core.env_server.http_server import create_app

from app.models import HospitalAction, HospitalObservation
from app.env import HospitalQueueEnvironment


app = create_app(
    env=HospitalQueueEnvironment,
    action_cls=HospitalAction,
    observation_cls=HospitalObservation,
    env_name="hospital_queue_env",
    max_concurrent_envs=4,
)