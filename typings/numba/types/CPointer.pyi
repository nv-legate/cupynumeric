# import numpy as np
from numba.core.types.abstract import Type

class CPointer(Type):
    def __init__(self, dtype: Type) -> None: ...
