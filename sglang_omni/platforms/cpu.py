from sglang.srt.platforms.cpu import CpuDeviceMixin

from sglang_omni.platforms.interface import OmniPlatform


class CPUOmniPlatform(CpuDeviceMixin, OmniPlatform):
    """CPU platform; uses the shared OmniPlatform hooks."""
