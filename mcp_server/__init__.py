'''MCP Server for PyClifford

This package provides a Model Context Protocol (MCP) server interface for the PyClifford library,
enabling AI agents to perform quantum computations and manipulations in the stabilizer formalism.
'''

__version__ = '0.1.0'

from .schemas import PauliTerm, Operator, CliffordUnitary
from .server import pauli_operator_product, operator_product, unitary_transform

__all__ = ['PauliTerm', 'Operator', 'CliffordUnitary', 'pauli_operator_product', 'operator_product'] 