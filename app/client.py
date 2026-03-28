"""Hospital Queue Env client wrapper."""

from openenv.core.env_client import EnvClient

from app.models import HospitalAction, HospitalObservation, HospitalState


class HospitalQueueEnv(EnvClient[HospitalAction, HospitalObservation, HospitalState]):
    _action_cls = HospitalAction
    _observation_cls = HospitalObservation
    _state_cls = HospitalState
