from fastmcp import FastMCP
from mcp_server.schemas import PauliTerm, Operator, CliffordUnitary
from typing import Literal
# Create the MCP server object
mcp = FastMCP('PyClifford Server')

@mcp.tool()
def pauli_operator_product(op1: PauliTerm, op2: PauliTerm) -> PauliTerm:
    ''' Compute the product (matrix multiplication) of two single Pauli terms.
    This tool accepts two PauliTerm objects and returns their product as a single PauliTerm, using the standard Pauli algebra rules.
    
    Examples:
        - Input: $(+i X_1) (Y_1)$
        - Input schema:
            - either use text field to specify PauliTerm:
                {
                    "op1": {"text": "+i X_1"},
                    "op2": {"text": "Y_1"}
                }
            - or use structured fields to specify PauliTerm:
                {
                    "op1": {"coefficient": 1j, "pauli_string": {"1": "X"}},
                    "op2": {"coefficient": 1, "pauli_string": {"1": "Y"}}
                }
        - Output schema:
            {
                "coefficient": "-1+0j",
                "pauli_string": {"1": "Z"},
                "text": "- Z_{1}"
            }
        - Output: $- Z_{1}$
    '''
    return PauliTerm.from_obj(op1.to_obj() @ op2.to_obj())

@mcp.tool()
def operator_product(op1: Operator, op2: Operator) -> Operator:
    ''' Compute the product (matrix multiplication) of two quantum operators (Pauli polynomials).
    This tool accepts two Operator objects (each a sum of Pauli terms) and returns their product as a new Operator, using the standard operator multiplication rules for Pauli polynomials.

    Examples:
        - Input: $(-0.5 Z_1 + 0.5 Z_2) (X_1)$
        - Input schema:
            - either use text field to specify Operator:
                {
                    "op1": {"text": "-0.5 Z_1 + 0.5 Z_2"},
                    "op2": {"text": "X_1"}
                }
            - or use structured fields to specify Operator:
                {
                    "op1": {
                        "terms": [
                            {"coefficient": -0.5, "pauli_string": {"1": "Z"}},
                            {"coefficient": 0.5, "pauli_string": {"2": "Z"}}
                        ]
                    },
                    "op2": {
                        "terms": [
                            {"coefficient": 1, "pauli_string": {"1": "X"}}
                        ]
                    }
                }
        - Output schema:
            {
                "terms": [
                    {
                        "coefficient": "0.5+0j",
                        "pauli_string": {"1": "X", "2": "Z"},
                        "text": "0.5 X_{1} Z_{2}"
                    },
                    {
                        "coefficient": "-0.5j",
                        "pauli_string": {"1": "Y"},
                        "text": "-0.5i Y_{1}"
                    }
                ],
                "text": "0.5 X_{1} Z_{2} - 0.5i Y_{1}"
            }
        - Output: $0.5 X_{1} Z_{2} - 0.5i Y_{1}$
    '''
    return Operator.from_obj(op1.to_obj() @ op2.to_obj())

@mcp.tool()
def unitary_transform(U: CliffordUnitary, O: Operator, direction: Literal["forward", "backward"] = "forward") -> Operator:
    ''' Apply a Clifford unitary to a quantum operator.
    This tool accepts a CliffordUnitary object and a quantum operator (Operator), and returns the result of applying the unitary to the operator.

    Examples:
        - Input: A Clifford unitary $U$ and a quantum operator $O$
        - Input schema:
            - either use text field to specify CliffordUnitary:
                {
                    "U": {"text": "X_1 -> + X_1 Y_2, Z_1 -> - X_1 Z_2, X_2 -> + X_2, Z_2 -> + Y_1 X_2"},
                    "O": {"text": "X_1"}
                }
            - or use structured fields to specify CliffordUnitary:
                {
                    "U": {
                        "clifford_map": {
                            "X_1": {"coefficient": 1, "pauli_string": {"1": "X", "2": "Y"}},
                            "Z_1": {"coefficient": -1, "pauli_string": {"1": "X", "2": "Z"}},
                            "X_2": {"coefficient": 1, "pauli_string": {"2": "X"}},
                            "Z_2": {"coefficient": 1, "pauli_string": {"1": "Y", "2": "X"}}
                        }
                    },
                    "O": {"text": "X_1"}
                }   
        - Output schema:
            {
                "terms": [
                    {"coefficient": "0.5+0j", "pauli_string": {"1": "X", "2": "Z"}, "text": "0.5 X_{1} Z_{2}"},
                    {"coefficient": "-0.5j", "pauli_string": {"1": "Y"}, "text": "-0.5i Y_{1}"}
                ],  
                "text": "0.5 X_{1} Z_{2} - 0.5i Y_{1}"
            }
        - Output: $0.5 X_{1} Z_{2} - 0.5i Y_{1}$
    '''
    gate = U.to_obj()
    op = O.to_obj()
    if direction == "forward":
        op, _ = gate.forward(op)
    else:
        op, _ = gate.backward(op)
    return Operator.from_obj(op)

# This is the main entry point for your server
if __name__ == "__main__":
    mcp.run()