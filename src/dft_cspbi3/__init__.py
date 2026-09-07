"""Caracterización DFT profunda con GPAW: el workflow de 26 pasos."""

__version__ = "0.6.1"
__author__ = "Contribuidores DFT-CsPbI3"

__all__ = [
    "StructureBuilder",
    "GPAWCalculatorFactory",
    "DFTWorkflow",
    "ScissorCorrection",
    "validation",
    "reporting",
]


def __getattr__(name: str):
    """Carga modulos pesados solo cuando se piden."""
    if name == "StructureBuilder":
        from .structure_builder import StructureBuilder
        return StructureBuilder
    if name == "GPAWCalculatorFactory":
        from .calculator_factory import GPAWCalculatorFactory
        return GPAWCalculatorFactory
    if name == "DFTWorkflow":
        from .workflow_manager import DFTWorkflow
        return DFTWorkflow
    if name == "ScissorCorrection":
        from .bandgap_correction import ScissorCorrection
        return ScissorCorrection
    if name in {"validation", "reporting"}:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)
