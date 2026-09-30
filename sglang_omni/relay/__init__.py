# SPDX-License-Identifier: Apache-2.0
"""Relay module for inter-stage data transfer.

Backends (shm, cuda_ipc, mooncake) register themselves with the relay
registry and are created through ``create_relay``.
"""

from sglang_omni.relay.base import Relay
from sglang_omni.relay.mooncake import MOONCAKE_AVAILABLE, MooncakeRelay

__all__ = [
    "Relay",
    "MooncakeRelay",
    "MOONCAKE_AVAILABLE",
]
