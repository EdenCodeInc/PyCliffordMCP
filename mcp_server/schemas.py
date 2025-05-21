from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Literal
from fastmcp.exceptions import ToolError
import re
import pyclifford as pc

class PauliTerm(BaseModel):
    '''Represents a single Pauli monomial: a complex scalar coefficient times a tensor product
    of single-qubit Pauli operators.

    - Also referred to as "PauliMonomial" in PyClifford, "PauliString" in Cirq, and "Pauli" in Qiskit.
    - A Pauli term is consist of a complex scalar coefficient multiplying a tensor product of single-qubit Pauli operators.
    - This model supports either structured input (`coefficient` and `pauli_string`) or text-based input (via `text`).
    - Pauli terms are the building blocks of operators: any operator can be expanded on the basis of Pauli strings.

    AI Agent Note:
        - When structured data is available, prefer specifying the `coefficient` and `pauli_string` fields directly for each Pauli term.
        - If you are provided with a raw text expression, or if parsing the expression into structured fields is ambiguous or error-prone, you may supply the entire expression to the `text` field.
        - The model will automatically parse the `text` input and populate the corresponding structured fields for you.
        - This approach is recommended for complex or lengthy operator expressions, or when you are uncertain about the correct structured representation.
    '''

    coefficient: complex | None = Field(
        default=1.0,
        description=(
            'The scalar coefficient of the Pauli term.\n\n'
            'Acceptable formats:\n'
            '- Complex number: "1+2j", "0.5j", "1+0j" (default)\n'
            '- Other number will be converted to complex: "2.0", "-0.5" -> "2.0+0.0j", "-0.5+0.0j"\n'
            '- Phase factors as strings: "+", "-", "+i", "-i", "i", "j"\n\n'
            'AI agent tip: You may use either "i" or "j" for the imaginary unit (validator can handle both).'
        )
    )

    pauli_string: dict[int, Literal['I', 'X', 'Y', 'Z']] | None = Field(
        default_factory=dict,
        description=(
            'Dictionary representing a tensor product of single-qubit Pauli operators.\n\n'
            'Keys: qubit indices (int)\n'
            'Values: Pauli operator names ("X", "Y", "Z", or "I")\n\n'
            'You may omit identity operators "I" to reduce verbosity; they are assumed by default.\n'
            'Examples:\n'
            '- {1: "X", 3: "Y"} -> applies X to qubit 1, Y to qubit 3\n'
            '- {0: "I", 1: "Z"} -> valid, but equivalent to {1: "Z"}'
        )
    )

    text: str | None = Field(
        default=None,
        description=(
            'Optional raw text representation of the Pauli term.\n\n'
            'If provided, this will be parsed automatically and will override the `coefficient` and `pauli_string` fields.\n'
            'Expected formats:\n'
            '- "-X_1 Z_2 Y_4" or "4+3i X(1) Z(2) Y(4)" (standard notation)\n'
            '- "iIXYIZ" or "-i X1Y2Z5" (compact notation)\n'
            r'- "0.5 \sigma_0^3 \sigma_1^2" or "1.2*\sigma^{01203}" (LaTeX notation)'
        )
    )

    @model_validator(mode='before')
    @classmethod
    def parse_from_text(cls, data: dict) -> dict:
        # parse a string like "+i X_1 Z_2 Y_4" into a dictionary for Pauli term model.
        # ---- helper function ----
        def _parse_coeff_idxops(text: str):
            # (1) Try stardard/explicit notation: X_1, Y3, Z_{2}, X(5)
            std_pattern = re.compile(r'([IXYZ])(?:_?\{?(\d+)\}?|\((\d+)\))')
            matches = list(std_pattern.finditer(text))
            if matches:
                coefficient = text[:matches[0].start()].strip()
                idxops = ((m.group(2) or m.group(3), m.group(1)) for m in matches)
                return coefficient, idxops
            # (2) Try LaTeX-style single-qubit operator: \sigma_1^{0}, \sigma^{x}_5, etc.
            latex_pattern = re.compile(
                r'\\[a-zA-Z]+(?:_\{?(\d+)\}?\^\{?([0-3xyzXYZI])\}?|'
                r'\^\{?([0-3xyzXYZI])\}?_\{?(\d+)\}?)'
            )
            matches = list(latex_pattern.finditer(text))
            if matches:
                coefficient = text[:matches[0].start()].strip()
                idxops = ((m.group(1) or m.group(4), m.group(2) or m.group(3)) for m in matches)
                return coefficient, idxops
            # (3) Try compact notation: XYZI
            compact_pattern = re.compile(r'([IXYZ])')
            matches = list(compact_pattern.finditer(text))
            if matches:
                coefficient = text[:matches[0].start()].strip()
                idxops = enumerate(m.group(1) for m in matches)
                return coefficient, idxops
            # (4) Try LaTeX index string: symbol^{1230}
            index_pattern = re.compile(r'(?:\\[a-zA-Z]+\^)?\{([0-3xyzIXYZ]+)\}')
            matches = list(index_pattern.finditer(text))
            if matches:
                coefficient = text[:matches[0].start()].strip()
                idxops = enumerate(matches[0].group(1))
                return coefficient, idxops
            return text.strip(), None
        # ---- main function ----
        # if 'text' is provided, it overrides and populates the other fields.
        if 'text' in data and data['text']:
            text = data.pop('text').strip()
            # remove LaTeX math mode formatting
            text = re.sub(r'\$(.*?)\$', r'\1', text)
            text = re.sub(r'\\\((.*?)\\\)', r'\1', text)
            text = re.sub(r'\\\[(.*?)\\\]', r'\1', text)
            try:
                coefficient, idxops = _parse_coeff_idxops(text)
            except Exception as e:
                raise ToolError(f"Failed to parse Pauli term from text '{text}': {e}")
            pauli_string = {}
            if idxops is not None:
                pauli_index_map = {'0':'I', '1':'X', '2':'Y', '3':'Z'}
                for idx, op in idxops:
                    op = pauli_index_map.get(op, op.upper())
                    if op != 'I':
                        pauli_string[int(idx)] = op
            parsed = {"coefficient": coefficient, "pauli_string": pauli_string}
            data['coefficient'] = parsed['coefficient']
            data['pauli_string'] = parsed['pauli_string']
        # the return statement must be placed outside if block
        return data

    @field_validator('coefficient', mode='before')
    @classmethod
    def parse_coefficient(cls, coefficient):
        if coefficient is None:
            return 1+0j
        if isinstance(coefficient, str):
            # remove all spaces and '*'
            coefficient_clean = coefficient.replace('*', '').replace(' ', '')
            common_phases = {
                '':     1+0j, '+':     1+0j, '-':     -1+0j,
                '1':    1+0j, '+1':    1+0j, '-1':    -1+0j,
                '1.0':  1+0j, '+1.0':  1+0j, '-1.0':  -1+0j,
                'i':    0+1j, '+i':    0+1j, '-i':    0-1j,
                'j':    0+1j, '+j':    0+1j, '-j':    0-1j,
                '1j':   0+1j, '+1j':   0+1j, '-1j':   0-1j,
                '1.0j': 0+1j, '+1.0j': 0+1j, '-1.0j': 0-1j
            }
            if coefficient_clean in common_phases:
                return common_phases[coefficient_clean]
            try:
                coeff = coefficient_clean.replace('i', 'j')
                if coeff.startswith('-(') and coeff.endswith(')'):
                    coeff = coeff[2:-1]
                    return -complex(coeff)
                else:
                    return complex(coeff)
            except ValueError:
                raise ToolError(
                    f'Invalid coefficient string: "{coefficient}". '
                    'Expected formats like "+i", "-1", "0.5", or "1+2j".'
                )
        if isinstance(coefficient, (float, int)):
            return complex(coefficient)
        return coefficient
    
    @model_validator(mode='after')
    def render_to_text(self):
        # provide a canonical text representation of the Pauli term
        phase_str_map = {1: '', 1j: '+i', -1: '-', -1j: '-i'}
        coeff = self.coefficient
        if coeff in phase_str_map:
            coeff = phase_str_map[coeff]
        else:
            real = coeff.real
            imag = coeff.imag
            if imag == 0:
                coeff = f'{real}'
            elif real == 0:
                if imag == 1:
                    coeff = '+i'
                elif imag == -1:
                    coeff = '-i'
                else:
                    coeff = f'{imag}i'
            else:
                sign = '+' if imag > 0 else '-'
                imag_abs = abs(imag)
                if imag_abs == 1:
                    imag_str = 'i'
                else:
                    imag_str = f'{imag_abs}i'
                coeff = f'({real}{sign}{imag_str})'
        if self.pauli_string:
            pauli_str = ' '.join(f'{op}_{{{idx}}}' for idx, op in self.pauli_string.items())
            self.text = f'{coeff} {pauli_str}'.strip()
        else:
            self.text = f'{coeff} I'.strip()
        return self
    
    def to_obj(self) -> pc.Pauli | pc.PauliMonomial:
        # Map phase factors to their corresponding phase indices
        phase_int_map = {1: 0, 1j: 1, -1: 2, -1j: 3}
        coeff = self.coefficient
        pauli = pc.pauli(self.pauli_string)
        if coeff in phase_int_map: # if coefficient is power of i 
            pauli.p = phase_int_map[coeff]
            return pauli # construct a Pauli object
        else: # otherwise, construct a PauliMonomial object
            return coeff * pauli
    
    @classmethod
    def from_obj(cls, obj) -> 'PauliTerm':
        def pauli_string(g: list[int]) -> dict:
            # convert a list of integers to a dictionary of operators
            pauli_string = {}
            N = g.shape[0]//2
            for i in range(N):
                x = g[2*i]
                z = g[2*i+1]
                if x == 1 and z == 0:
                    pauli_string[i] = 'X'
                elif x == 1 and z == 1:
                    pauli_string[i] = 'Y'
                elif x == 0 and z == 1:
                    pauli_string[i] = 'Z'
                # skip I operator (x=0, z=0)
            return pauli_string
        if isinstance(obj, pc.PauliMonomial):
            return cls(coefficient=obj.c * 1j**obj.p, pauli_string=pauli_string(obj.g))
        if isinstance(obj, pc.Pauli): # Don't move it before PauliMonomial check
            return cls(coefficient=1j**obj.p, pauli_string=pauli_string(obj.g))
        else:
            raise ValueError(f"Unsupported object type: {type(obj)}")

    def to_operator(self):
        '''Promote this PauliTerm to an Operator containing only itself.'''
        return Operator(terms=[self])

class Operator(BaseModel):
    '''Represents a generic quantum operator as a linear combination of Pauli strings.

    - Also referred to as "PauliPolynomial" in PyClifford, "PauliSum" in Cirq, and "PauliSumOp" in Qiskit.
    - An operator is a sum of Pauli terms, each of which is a complex scalar coefficient multiplying a tensor product of single-qubit Pauli operators.
    - This model supports either structured input (`terms`) or text-based input (via `text`).
    - Operators can be used to represent Hamiltonians, observables, or any general operator expression.

    AI Agent Note:
        - When structured data is available, represent each term as a `PauliTerm` object using the `coefficient` and `pauli_string` fields.
        - Collect all terms in a list and assign it to the `terms` field to form the complete operator.
        - If you are provided with a raw text expression, or if parsing the expression into structured fields is ambiguous or error-prone, you may supply the entire operator expression to the `text` field.
        - The model will automatically parse the `text` input and populate the `terms` field for you.
        - This approach is recommended for complex, lengthy, or ambiguous operator expressions, or when you are uncertain about the correct structured representation.

    Examples:
        - "+i X_1 Y_2"         # One term: +i times X on qubit 1 and Y on qubit 2
        - "-0.5 Z_1 + 0.5 Z_2" # Two terms: -0.5 times Z on qubit 1, plus 0.5 times Z on qubit 2
    '''

    terms: list[PauliTerm] | None = Field(
        default_factory=list,
        description=(
            'List of PauliTerm objects, each representing a scalar-multiplied Pauli string.\n'
            'Each term may be provided either with structured input (`coefficient` and `pauli_string`) or as text (via `text`).\n\n'
            'Examples:\n'
            '- [{ "text": "+i X_1 Z_2" }]\n'
            '- [{ "coefficient": -1, "pauli_string": {1: "Z", 2: "Z"} }]'
        )
    )

    text: str | None = Field(
        default=None,
        description=(
            'Optional raw text representation of the entire operator (sum of Pauli terms).\n\n'
            'If provided, this string will be parsed automatically to populate the `terms` field with a list of PauliTerm objects.\n'
            'Expected formats:\n'
            '- Each term should start with a \'+\' or \'-\' sign (the sign belongs to the term).\n'
            '- Terms may be separated by spaces, but spaces are not required or reliable.\n'
            '- Each term can use any notation supported by PauliTerm (standard, compact, or LaTeX-style).\n'
            '- Example: \'+i X_1 Y_2 -0.5 Z_1 + 0.5 Z_2\'\n'
            '- Example: \'-X1Y2Z3+Y1Z2-X3\'\n'
            '- Example: \'1.2*\\sigma^{01203} - 0.5*X_1Y_2\''
        )
    )

    @model_validator(mode='before')
    @classmethod
    def parse_from_text(cls, data: dict) -> dict:
        if 'text' in data and data['text']:
            text = data.pop('text').strip()
            # remove LaTeX math mode formatting
            text = re.sub(r'\$(.*?)\$', r'\1', text)
            text = re.sub(r'\\\((.*?)\\\)', r'\1', text)
            text = re.sub(r'\\\[(.*?)\\\]', r'\1', text)
            # split the text into terms
            try:
                terms = []
                i = 0
                start = 0
                depth = 0
                while i < len(text):
                    c = text[i]
                    if c == '(':
                        depth += 1
                    elif c == ')':
                        depth -= 1
                    elif c in '+-' and i != start and depth == 0:
                        # Split here, but not at the very start, and only if not inside parentheses
                        terms.append(text[start:i].strip())
                        start = i
                    i += 1
                # Add the last term
                if start < len(text):
                    terms.append(text[start:].strip())
                terms = [PauliTerm(text = term) for term in terms if term != '']
            except Exception as e:
                raise ToolError(f"Failed to parse operator from text '{text}': {e}")
            data['terms'] = terms
        # the return statement must be placed outside if block
        return data
        
    @model_validator(mode='after')
    def render_to_text(self):
        if not self.terms:
            self.text = '0'
        else:
            self.text = str(self.terms[0].text)
            for term in self.terms[1:]:
                t = str(term.text)
                if t.lstrip().startswith('-'):
                    self.text += ' - ' + t.lstrip('-').lstrip()
                else:
                    self.text += ' + ' + t.lstrip('+').lstrip()
            self.text = self.text.strip()
        return self
        
    def to_obj(self) -> pc.PauliPolynomial:
        poly = pc.pauli_zero(0)
        for term in self.terms:
            poly += term.to_obj()
        return poly
    
    @classmethod
    def from_obj(cls, obj) -> 'Operator':
        if isinstance(obj, pc.PauliPolynomial):
            terms = [PauliTerm.from_obj(term) for term in obj]
            return cls(terms=terms)
        if isinstance(obj, (pc.Pauli, pc.PauliMonomial)):
            return cls.from_obj(obj.as_polynomial())
        else:
            raise ValueError(f"Unsupported object type: {type(obj)}")

class CliffordUnitary(BaseModel):
    '''Represents a Clifford unitary transformation specifiedby its action on Pauli operators.

    - A Clifford unitary $U$ is defined by its transformation (conjugation action) on any operator $O$ as $O -> U O U^H$.
    - The action of $U$ is fully specified by its effect on the Pauli group generators (e.g., $X_i$, $Z_i$ for $i=1,...,N$).
    - The forward map is a dictionary mapping each generator (as a string representation of a PauliTerm) 
      to its image under unitary transformation (as a PauliTerm).

    AI Agent Note:
        - The `clifford_map` field should be a dictionary where keys are string representations of Pauli group generators and values are PauliTerm objects.
        - The `text` field provides a human-readable or text-based description of the Clifford unitary, e.g., a table of generator mappings.
        - This model is useful for representing, serializing, and communicating Clifford unitaries in a structured way.

    Example:
        Generator forward map ($O -> U O U^H$):
            X_1 -> + X_1 Y_2
            Z_1 -> - X_1 Z_2
            X_2 -> + X_2
            Z_2 -> + Y_1 X_2
        Input schema:
            - either use text field to specify Clifford unitary:
                {
                    "text": "X_1 -> + X_1 Y_2, Z_1 -> - X_1 Z_2, X_2 -> + X_2, Z_2 -> + Y_1 X_2"
                }
            - or use dictionary of text representations of Pauli strings to specify Clifford unitary:
                {
                    "clifford_map": {
                        "X_1": "+ X_1 Y_2",
                        "Z_1": "- X_1 Z_2",
                        "X_2": "+ X_2",
                        "Z_2": "+ Y_1 X_2"
                    }
                }
            - or use dictionary of structured PauliTerm objects to specify Clifford unitary:
                {
                    "clifford_map": {
                        "X_1": {"coefficient": 1, "pauli_string": {"1": "X", "2": "Y"}},
                        "Z_1": {"coefficient": -1, "pauli_string": {"1": "X", "2": "Z"}},
                        "X_2": {"coefficient": 1, "pauli_string": {"2": "X"}},
                        "Z_2": {"coefficient": 1, "pauli_string": {"1": "Y", "2": "X"}}
                    }
                }
    '''

    clifford_map: dict[str, PauliTerm] = Field(
        default_factory=dict,
        description=(
            "Dictionary representing the forward action of the Clifford unitary on Pauli generators. "
            "Each key is a string representation of a Pauli group generator (e.g., 'X1', 'Z_2', 'X{3}', 'Z(4)'). "
            "Each value is the image PauliTerm under conjugation."
        )
    )

    text: str | None = Field(
        default=None,
        description=(
            "Optional human-readable text-based expression of the Clifford unitary, "
            "such as a table of generator mappings."
        )
    )

    @model_validator(mode='before')
    @classmethod
    def parse_from_text(cls, data):
        if 'text' in data and data['text']:
            text = data.pop('text')
            # Split into mapping entries (comma, \\, or newline as separator)
            entries = re.split(r',|\\\\|\n', text)
            clifford_map = {}
            for entry in entries:
                entry = entry.strip()
                if entry == '':
                    continue
                # Split key and value by mapping arrow (->, :, \to, \mapsto, \rightarrow)
                m = re.split(r'\s*(?:->|:|\\to|\\mapsto|\\rightarrow)\s*', entry, maxsplit=1)
                if len(m) < 2:
                    raise ToolError(f"Invalid mapping entry: '{entry}'. Expected format like 'X_1 -> + X_1 Y_2'.")
                key = m[0].strip()
                value = m[1].strip()
                clifford_map[key] = value
            data['clifford_map'] = clifford_map
        return data

    @field_validator('clifford_map', mode='before')
    @classmethod
    def parse_clifford_map(cls, clifford_map):
        key_pattern = re.compile(r"[XZ]_\{\d+\}")
        new_map = {}
        for key, value in clifford_map.items():
            # Parse the key as a PauliTerm
            key_term = PauliTerm(text=key)
            # Extract all matches for single Pauli operator
            matches = key_pattern.findall(key_term.text)
            if len(matches) != 1:
                raise ToolError(f"Clifford map key '{key}' must contain exactly one single-qubit Pauli operator (X_{{int}} or Z_{{int}}). Found: {matches}")
            key_str = matches[0]
            # If key has a phase, invert it and apply to value
            phase = key_term.coefficient
            if isinstance(value, str):
                value_term = PauliTerm(text=value)
            else:
                value_term = value
            # If key phase is not 1, invert and apply to value
            if phase != 1:
                # Multiply value's coefficient
                value_coeff = value_term.coefficient / phase
                value_term = value_term.model_copy(update={"coefficient": value_coeff})
            new_map[key_str] = value_term
        return new_map
    
    @model_validator(mode='after')
    def render_to_text(self):
        self.text = ''  # Initialize as empty string
        for key, value in self.clifford_map.items():
            self.text += f'{key} -> {value.text}, '
        self.text = self.text.rstrip(', ')
        return self
    
    def to_obj(self) -> pc.CliffordGate:
        qubits = set()
        for key in self.clifford_map.keys():
            m = re.match(r'[XZ]_\{(\d+)\}', key)
            if m:
                qubits.add(int(m.group(1)))
        qubits = sorted(qubits)
        if not qubits:
            return pc.CliffordGate()
        i0 = min(qubits)
        ops = []
        for i in qubits:
            for op in ("X", "Z"):
                key = f"{op}_{{{i}}}"
                value = self.clifford_map.get(key, PauliTerm(text=key))
                value.pauli_string = {int(i) - i0: op for i, op in value.pauli_string.items()}
                ops.append(value.to_obj())
        ops = pc.paulis(ops)
        clifford_map = pc.CliffordMap(ops.gs, ops.ps)
        gate = pc.CliffordGate(*qubits)
        gate.set_forward_map(clifford_map)
        return gate
        
    @classmethod
    def from_obj(cls, obj) -> 'CliffordUnitary':
        def clifford_map(ops: pc.PauliList, i0: int = 0) -> dict:
            print(f'i0 = {i0}')
            keys, values = [], []
            for i in range(ops.N):
                for op in ("X", "Z"):
                    keys.append(f"{op}_{{{i + i0}}}")
            for op in ops:
                value = PauliTerm.from_obj(op)
                pauli_string = {int(i) + i0: op for i, op in value.pauli_string.items()}
                value = value.model_copy(update={"pauli_string": pauli_string})
                values.append(value)
            return {key: value for key, value in zip(keys, values)}
        if isinstance(obj, pc.CliffordGate):
            return cls(clifford_map=clifford_map(obj.forward_map, i0 =min(obj.qubits)))
        elif isinstance(obj, pc.CliffordMap):
            return cls(clifford_map=clifford_map(obj))
        else:
            raise ValueError(f"Unsupported object type: {type(obj)}")