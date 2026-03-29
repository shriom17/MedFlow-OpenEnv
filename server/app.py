"""FastAPI app for the Hospital Queue Management Environment."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from app.models import HospitalAction, HospitalObservation
from server.hospital_environment import HospitalQueueEnvironment

app = create_app(
    env=HospitalQueueEnvironment,
    action_cls=HospitalAction,
    observation_cls=HospitalObservation,
    env_name="hospital_queue_env",
    max_concurrent_envs=4,
)
