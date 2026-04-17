"""High-level inference runtime.

Consumes the odte.kernels CPU stubs today and the Hopper-native kernels on
an H100 box. Provides one import target for the rest of the stack:

    from odte.runtime import PersistentInferenceStub, RuntimeBench
"""
from .persistent_inference import PersistentInferenceStub
from .bench import RuntimeBench
