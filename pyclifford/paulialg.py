import numpy
from .utils import (
    qbit_int_type, indc_int_type, cplx_type, indc_mod,
    ipow, clifford_rotate, pauli_transform,
    batch_dot, aggregate, merge_qubits, as_qubits)
from functools import reduce
from scipy.linalg import logm, expm

class Pauli(object):
    '''Represents a Pauli operator.

    Parameters:
    g: int (2*N) - a Pauli string in binary repr.
    p: int - phase indicator (i power).'''
    def __init__(self, g, p = None, qubits = None, **kwargs):
        self.g = g
        self.p = 0 if p is None else p
        if qubits is None:
            self.qubits = as_qubits(self.N)
        else:
            if len(qubits) != self.N:
                raise ValueError(f"Length of qubits {len(qubits)} does not match Pauli string length {self.N}.")
            self.qubits = as_qubits(qubits)
        # other kwargs ignored, to absorb subclass-specific kwargs

    def embed_qubits(self, qubits, inds = None):
        '''Embed a Pauli operator into a larger set of qubits (inplace).
        
        Parameters:
        qubits: int (N') - larger target set of qubits.
            (User must ensure that qubits is sorted and self.qubits is its subset.)
        inds: int (N) - index of current qubits in target qubits set.

        Returns:
        self: Pauli operator with embedded qubits.'''
        qubits = as_qubits(qubits)
        if numpy.array_equal(qubits, self.qubits):
            return self
        if inds is None:
            inds = numpy.searchsorted(qubits, self.qubits)
        g = numpy.zeros(2*len(qubits), dtype=indc_int_type)
        inds2 = numpy.column_stack([2*inds, 2*inds+1]).flatten()
        g[inds2] = self.g
        self.g = g
        self.qubits = qubits
        return self
    
    def prune_qubits(self):
        '''Prune the qubits of a Pauli operator (inplace).
        
        Returns:
        self: Pauli operator with pruned qubits.'''
        mask = (self.g[0::2] != 0) | (self.g[1::2] != 0)
        if not numpy.any(mask):
            self.g = numpy.zeros(0, dtype=indc_int_type)
            self.qubits = as_qubits(0)
            return self
        self.g = self.g[numpy.repeat(mask, 2)]
        self.qubits = self.qubits[mask]
        return self

    def _pauli_str(self):
        # Helper method to convert Pauli operator to string representation.
        pauli_str_map = {(1,0): ' X', (1,1): ' Y', (0,1): ' Z'}
        txt = ''
        for i in range(self.N):
            xz = tuple(self.g[2*i:2*i+2])
            if xz != (0,0):
                txt += f'{pauli_str_map[xz]}{self.qubits[i]}'
        if txt == '':
            txt = 'I'
        return txt.strip()

    def __repr__(self):
        # interprete phase factor
        phase_str_map = {0: '+ ', 1: '+i ', 2: '- ', 3: '-i '}
        txt = phase_str_map[self.p] + self._pauli_str()
        return txt.strip()

    @property
    def N(self): # number of qubits
        return self.g.shape[0]//2
    
    def __neg__(self):
        return type(self)(self.g, indc_mod(self.p + 2, 4), qubits=self.qubits)

    def __rmul__(self, c):
        if c == 1:
            return self
        elif c == 1j:
            return type(self)(self.g, indc_mod(self.p + 1, 4), qubits=self.qubits)
        elif c == -1:
            return type(self)(self.g, indc_mod(self.p + 2, 4), qubits=self.qubits)
        elif c == -1j:
            return type(self)(self.g, indc_mod(self.p + 3, 4), qubits=self.qubits)
        else: # upgrade to PauliMonomial
            return c * self.as_monomial()

    def __truediv__(self, other):
        return (1/other) * self

    def __add__(self, other):
        return self.as_polynomial() + other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        if isinstance(other, Pauli):
            # If qubit sets differ, embed both to the union
            if not numpy.array_equal(self.qubits, other.qubits):
                qubits, inds1, inds2 = merge_qubits(self.qubits, other.qubits)
                # avoid in-place modification
                me = self.copy().embed_qubits(qubits, inds1)
                other = other.copy().embed_qubits(qubits, inds2)
            else:
                me = self
            p = indc_mod(me.p + other.p + ipow(me.g, other.g), 4)
            g = indc_mod(me.g + other.g, 2)
            return Pauli(g, p, me.qubits)
        elif isinstance(other, (PauliMonomial, PauliPolynomial)):
            return self.as_polynomial() @ other.as_polynomial()
        else: 
            raise NotImplementedError('matmul is not implemented for between {} and {}'.format(type(self).__name__, type(other).__name__))

    def trace(self):
        # normalized trace: 1 for identity operator, 0 otherwise
        if numpy.sum(self.g) == 0:
            return 2**self.N
        else:
            return 0
        
    def inner(self, other):
        return self.as_polynomial().inner(other.as_polynomial())

    def weight(self):
        return numpy.sum(numpy.sum(self.g.reshape(self.N, 2), -1) != 0)

    def copy(self):
        return type(self)(self.g.copy(), self.p, qubits=self.qubits)

    def as_monomial(self):
        '''cast a Pauli operator to a Pauli monomial assuming coefficient = 1'''
        return PauliMonomial(self.g, self.p, qubits=self.qubits)

    def as_polynomial(self):
        '''cast a Pauli operator to a Pauli polynomial'''
        return self.as_monomial().as_polynomial()

    def as_list(self):
        '''cast a Pauli operator to a Pauli list'''
        gs = numpy.expand_dims(self.g, 0)
        ps = numpy.array([self.p], dtype=indc_int_type)
        return PauliList(gs, ps, qubits=self.qubits)

    def rotate_by(self, generator):
        # see PauliList.rotate_by for details
        result = self.as_list().rotate_by(generator)
        self.g = result.gs[0]
        self.p = result.ps[0]
        self.qubits = result.qubits
        return self

    def transform_by(self, clifford_map):
        # see PauliList.transform_by for details
        result = self.as_list().transform_by(clifford_map)
        self.g = result.gs[0]
        self.p = result.ps[0]
        self.qubits = result.qubits
        return self

    def inverse(self):
        return self.as_monomial().inverse()
    
    def __pow__(self, n):
        return self.as_polynomial().__pow__(n)

    def sqrt(self):
        return self.as_polynomial().sqrt()
    
    def exp(self):
        return self.as_polynomial().exp()
    
    def log(self):
        return self.as_polynomial().log()

    def to_numpy(self, qubits=None):
        """Convert Pauli operator to numpy array representation on the full system.

        Parameters:
            qubits: array-like or None
                If provided, specifies the set of qubits to be included.
                If None, uses self.qubits.
                If 'all', uses all qubits in the system.

        Returns:
            np.ndarray: The matrix representation.
        """
        sigma = numpy.array([
            [[1, 0], [0, 1]],      # I (00)
            [[0, 1], [1, 0]],      # X (10)
            [[0, -1j], [1j, 0]],   # Y (11)
            [[1, 0], [0, -1]]      # Z (01)
        ], dtype=cplx_type)
        # Handle empty qubits support case
        if qubits is None:
            qubits = self.qubits
        elif qubits == 'all':
            qubits = as_qubits(numpy.max(self.qubits) + 1)
        else:
            qubits = as_qubits(qubits)
        # Handle empty Pauli operator case
        if len(qubits) == 0:
            return numpy.ones((1, 1), dtype=cplx_type)
        # Build list of matrices for tensor product
        matrices = []
        qubit_map = {q: i for i, q in enumerate(self.qubits)}
        for q in qubits:
            if q in qubit_map:
                i = qubit_map[q]
                x = self.g[2*i]
                z = self.g[2*i+1]
                idx = x + 3*z - 2*x*z
            else:
                idx = 0  # Identity
            matrices.append(sigma[idx])
        result = reduce(numpy.kron, matrices)
        # Apply phase factor
        return (1j)**(self.p) * result

class PauliList(object):
    '''Represents a list of Pauli operators.

    Parameters:
    gs: int (L, 2*N) - array of Pauli strings in binary repr.
    ps: int (L) - array of phase indicators (i powers).'''
    def __init__(self, gs, ps=None, qubits=None, **kwargs):
        self.gs = gs
        self.ps = numpy.zeros(self.L, dtype=indc_int_type) if ps is None else ps
        if qubits is None:
            self.qubits = as_qubits(self.N)
        else:
            if len(qubits) != self.N:
                raise ValueError(f"Length of qubits {len(qubits)} does not match Pauli string length {self.N}.")
            self.qubits = as_qubits(qubits)
        # kwargs ignored, in case subclass-specific arguments passed up

    def embed_qubits(self, qubits, inds = None):
        '''Embed a Pauli list into a larger set of qubits (inplace).
        
        Parameters:
        qubits: int (N) - larger targetset of qubits.
            (User must ensure that qubits is sorted and self.qubits is its subset.)
        inds: int (N) - index of current qubits in target qubits set.

        Returns:
        self: Pauli list with embedded qubits.'''
        qubits = as_qubits(qubits)
        if numpy.array_equal(qubits, self.qubits):
            return self
        if inds is None:
            inds = numpy.searchsorted(qubits, self.qubits)
        gs = numpy.zeros((self.L, 2*len(qubits)), dtype=self.gs.dtype)
        inds2 = numpy.column_stack([2*inds, 2*inds+1]).flatten()
        gs[:,inds2] = self.gs
        self.gs = gs
        self.qubits = qubits
        return self
    
    def prune_qubits(self):
        '''Prune the qubits of a Pauli list (inplace).
        
        Returns:
        self: Pauli list with pruned qubits.'''
        mask = ((self.gs[:, 0::2] != 0) | (self.gs[:, 1::2] != 0)).any(axis=0)
        if not numpy.any(mask):
            self.gs = numpy.zeros((self.L, 0), dtype=indc_int_type)
            self.qubits = as_qubits(0)
            return self
        self.gs = self.gs[:, numpy.repeat(mask, 2)]
        self.qubits = self.qubits[mask]
        return self

    def __repr__(self):
        if not self:
            return '[]'
        return '[' + ',\n '.join([repr(pauli) for pauli in self]) + ']'

    def __len__(self):
        return self.L

    @property
    def L(self):
        return self.gs.shape[0]

    @property
    def N(self):
        return self.gs.shape[1]//2

    def __getitem__(self, item):
        if isinstance(item, (int, numpy.integer)):
            return Pauli(self.gs[item], self.ps[item], self.qubits)
        return PauliList(self.gs[item], self.ps[item], self.qubits)

    def __add__(self, other):
        # join two PauliLists
        if isinstance(other, PauliList):
            # If qubit sets differ, embed both to the union
            if not numpy.array_equal(self.qubits, other.qubits):
                qubits, inds1, inds2 = merge_qubits(self.qubits, other.qubits)
                # avoid in-place modification
                me = self.copy().embed_qubits(qubits, inds1)
                other = other.copy().embed_qubits(qubits, inds2)
            else:
                me = self
            gs = numpy.concatenate([me.gs, other.gs])
            ps = numpy.concatenate([me.ps, other.ps])
            return type(self)(gs, ps, qubits=qubits)
        else:
            raise TypeError(f"Unsupported operand type for +: '{type(self).__name__}' and '{type(other).__name__}'")

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return type(self)(self.gs, indc_mod(self.ps + 2, 4), qubits=self.qubits)

    def __truediv__(self, other):
        return (1/other) * self

    def __rmul__(self, c):
        if c == 1:
            return self
        elif c == 1j:
            return type(self)(self.gs, indc_mod(self.ps + 1, 4), qubits=self.qubits)
        elif c == -1:
            return type(self)(self.gs, indc_mod(self.ps + 2, 4), qubits=self.qubits)
        elif c == -1j:
            return type(self)(self.gs, indc_mod(self.ps + 3, 4), qubits=self.qubits)
        else: # upgrade to PauliPolynomial
            raise NotImplementedError('multiplication is not defined for {} when factor is not 1, -1, 1j, -1j.'.format(type(self).__name__))

    def trace(self):
        # normalized trace: 1 for identity operator, 0 otherwise
        return numpy.where(numpy.sum(self.gs, -1) == 0, 1, 0)

    def weight(self):
        return numpy.sum(numpy.sum(self.gs.reshape(self.L, self.N, 2), -1) != 0, -1)

    def copy(self):
        return type(self)(self.gs.copy(), self.ps.copy(), qubits=self.qubits.copy())

    def as_polynomial(self):
        return PauliPolynomial(self.gs, self.ps, qubits=self.qubits)
    
    def rotate_by(self, generator):
        '''Perform Clifford rotation by Pauli generator (in-place).
            O ->  U^dag O U, with U = exp(i pi/4 g), given g the generator.
        
        Parameters:
        generator: Pauli - Pauli generator.

        Returns:
        self: PauliList - Pauli list with rotated Pauli operators.'''
        if not numpy.array_equal(self.qubits, generator.qubits):
            # merge and embed qubit support
            qubits, inds1, inds2 = merge_qubits(self.qubits, generator.qubits)
            self.embed_qubits(qubits, inds1) # in-place modification
        # if generator acts on a strict subset of qubits, rotate only the subset. 
        if len(generator.qubits) < len(self.qubits):
            inds = numpy.column_stack([2*inds2, 2*inds2+1]).flatten()
            self.gs[:,inds], self.ps = clifford_rotate(
                generator.g, generator.p, self.gs[:,inds], self.ps)
        else:
            clifford_rotate(generator.g, generator.p, self.gs, self.ps)
        return self

    def transform_by(self, clifford_map):
        '''Perform Clifford transformation by Clifford map (in-place).
        
        Parameters:
        clifford_map: CliffordMap - Clifford map of the unitary transformation.

        Returns:
        self: PauliList - Pauli list with transformed Pauli operators.'''
        # merge and embed qubit support
        if not numpy.array_equal(self.qubits, clifford_map.qubits):
            qubits, inds1, inds2 = merge_qubits(self.qubits, clifford_map.qubits)
            self.embed_qubits(qubits, inds1) # in-place modification
        # if clifford_map acts on a strict subset of qubits, transform only the subset. 
        if len(clifford_map.qubits) < len(self.qubits):
            inds = numpy.column_stack([2*inds2, 2*inds2+1]).flatten()
            self.gs[:,inds], self.ps = pauli_transform(
                self.gs[:,inds], self.ps, clifford_map.gs, clifford_map.ps)
        else:
            self.gs, self.ps = pauli_transform(
                self.gs, self.ps, clifford_map.gs, clifford_map.ps)
        return self

    def append(self, other):
        # append a Pauli operator to the end of the list (in-place)
        if isinstance(other, Pauli):
             # If qubit sets differ, embed both to the union
            if not numpy.array_equal(self.qubits, other.qubits):
                qubits, inds1, inds2 = merge_qubits(self.qubits, other.qubits)
                self.embed_qubits(qubits, inds1)
                other = other.embed_qubits(qubits, inds2)
            other = other.as_list()
            self.gs = numpy.concatenate([self.gs, other.gs])
            self.ps = numpy.concatenate([self.ps, other.ps])
        else:
            raise TypeError(f"Unsupported operand type for +: '{type(self).__name__}' and '{type(other).__name__}'")

    def to_numpy(self, qubits = None):
        """Convert list of Pauli operators to numpy array representations in batch.
        Returns a (L, 2^N, 2^N) array where L is the number of Pauli operators."""
        # Define all Pauli matrices as a single 4x2x2 array
        sigma = numpy.array([
            [[1, 0], [0, 1]],      # I (00)
            [[0, 1], [1, 0]],      # X (10)
            [[0, -1j], [1j, 0]],   # Y (11)
            [[1, 0], [0, -1]]      # Z (01)
        ], dtype=cplx_type)
        # Process qubits
        if qubits is None:
            qubits = self.qubits
        elif qubits == 'all':
            qubits = as_qubits(numpy.max(self.qubits) + 1)
        else:
            qubits = as_qubits(qubits)
        # Handle empty Pauli operator case
        if len(qubits) == 0:
            return numpy.ones((self.L, 1, 1), dtype=cplx_type)
        # For each qubit position, get the corresponding Pauli matrices for all operators
        mats = []
        qubit_map = {q: i for i, q in enumerate(self.qubits)}
        for q in qubits:
            if q in qubit_map:
                i = qubit_map[q]
                x = self.gs[:,2*i]
                z = self.gs[:,2*i+1]
                idx = x + 3*z - 2*x*z
            else:
                idx = numpy.zeros(self.L, dtype=indc_int_type)  # Identity
            mats.append(sigma[idx])
        # Compute tensor product for all operators simultaneously
        def batched_kron(a, b):
            # a: (L, m, m), b: (L, n, n)
            return numpy.einsum('mij, mkl -> mikjl', a, b).reshape(a.shape[0], a.shape[1]*b.shape[1], a.shape[2]*b.shape[2])
        raw_reps = reduce(batched_kron, mats)
        # Apply phases
        reps = (1j)**(self.ps[:,None,None]) * raw_reps
        return reps

class PauliMonomial(Pauli):
    '''Represent a Pauli operator with a coefficient.

    Parameters:
    g: int (2*N) - a Pauli string in binary repr.
    p: int - phase indicator (i power).
    c: comlex - coefficient.'''
    def __init__(self, *args, **kwargs):
        # extract c and remove it from kwargs, if present.
        self.c = kwargs.pop('c', 1.+0.j) # default coefficient 1.0+0.0j
        super(PauliMonomial, self).__init__(*args, **kwargs)

    def __repr__(self):
        # interprete coefficient
        c = self.c * 1j**self.p
        phase_str_map = {1+0j: '+ ', 1j: '+i ', -1+0j: '- ', -1j: '-i '}
        if c == 0.:
            return '0'
        if c in phase_str_map:
            return phase_str_map[c] + self._pauli_str()
        if c.imag == 0.:
            c = c.real
            if c.is_integer():
                txt = '{:d} '.format(int(c))
            else: 
                txt = '{:.2f} '.format(c)
        else:
            txt = '({:.2f}) '.format(c).replace('j', 'i')
        txt += self._pauli_str()
        return txt.strip()

    def __neg__(self):
        return PauliMonomial(self.g, self.p, c=-self.c, qubits=self.qubits)

    def __rmul__(self, c):
        return PauliMonomial(self.g, self.p, c=c * self.c, qubits=self.qubits)

    def __truediv__(self, other):
        return (1/other) * self

    def __add__(self, other):
        return self.as_polynomial() + other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        if isinstance(other, (Pauli, PauliMonomial, PauliPolynomial)):
            return self.as_polynomial() @ other.as_polynomial()
        else:
            raise NotImplementedError('matmul is not implemented for between {} and {}'.format(type(self).__name__, type(other).__name__))

    def trace(self):
        # normalized trace: 1 for identity operator, 0 otherwise
        return self.c * super(PauliMonomial, self).trace()

    def inner(self, other):
        return self.as_polynomial().inner(other.as_polynomial())

    def copy(self):
        return PauliMonomial(self.g.copy(), self.p, c=self.c, qubits=self.qubits.copy())
    
    def as_list(self):
        '''cast the Pauli monomial to a single-term Pauli list
           ignore the coefficient '''
        gs = numpy.expand_dims(self.g, 0)
        ps = numpy.array([self.p], dtype=indc_int_type)
        return PauliList(gs, ps, qubits=self.qubits)
    
    def as_polynomial(self):
        '''cast the Pauli monomial to a single-term Pauli polynomial'''
        gs = numpy.expand_dims(self.g, 0)
        ps = numpy.array([self.p], dtype=indc_int_type)
        cs = numpy.array([self.c], dtype=cplx_type)
        return PauliPolynomial(gs, ps, cs=cs, qubits=self.qubits)

    def inverse(self):
        return Pauli(self.g, qubits=self.qubits)/(self.c * 1j**self.p)
    
    def __pow__(self, n):
        return self.as_polynomial().__pow__(n)
    
    def sqrt(self):
        return self.as_polynomial().sqrt()
    
    def exp(self):  
        return self.as_polynomial().exp()
    
    def log(self):
        return self.as_polynomial().log()

    def to_numpy(self, qubits=None):
        """Convert Pauli monomial to numpy array representation."""
        return self.c * super().to_numpy(qubits)

class PauliPolynomial(PauliList):
    '''Represent a linear combination of Pauli operators.

    Parameters:
    gs: int (L, 2*N) - array of Pauli strings in binary repr.
    ps: int (L) - array of phase indicators (i powers).
    cs: comlex (L) - coefficients.'''
    def __init__(self, *args, **kwargs):
        # extract cs and remove it from kwargs, if present.
        self.cs = kwargs.pop('cs', None) 
        super(PauliPolynomial, self).__init__(*args, **kwargs)
        if self.cs is None:
            self.cs = numpy.ones(self.ps.shape, dtype=cplx_type)

    def __repr__(self):
        if self.L == 0:
            return '0'
        txt = repr(self[0]).lstrip('+').lstrip()
        for term in self[1:]:
            txt_term = repr(term)
            if txt_term.lstrip().startswith('-'):
                txt += ' - ' + txt_term.lstrip('-').lstrip()
            else:
                txt += ' + ' + txt_term.lstrip('+').lstrip()
        return txt

    def __getitem__(self, item):
        if isinstance(item, (int, numpy.integer)):
            return PauliMonomial(self.gs[item], self.ps[item], c=self.cs[item], qubits=self.qubits)
        return PauliPolynomial(self.gs[item], self.ps[item], cs=self.cs[item], qubits=self.qubits)

    def __neg__(self):
        return PauliPolynomial(self.gs, self.ps, cs=-self.cs, qubits=self.qubits)

    def __rmul__(self, c):
        return PauliPolynomial(self.gs, self.ps, cs=c * self.cs, qubits=self.qubits)

    def __truediv__(self, other):
        return (1/other) * self

    def __add__(self, other):
        if not isinstance(other, PauliPolynomial):
            if isinstance(other, (PauliMonomial, Pauli, PauliList)):
                other = other.as_polynomial()
            else: # otherwise assuming other is a number
                other = other * pauli_identity()
        # If qubit sets differ, embed both to the union
        if not numpy.array_equal(self.qubits, other.qubits):
            qubits, inds1, inds2 = merge_qubits(self.qubits, other.qubits)
            # avoid in-place modification
            me = self.copy().embed_qubits(qubits, inds1)
            other = other.copy().embed_qubits(qubits, inds2)
        else:
            me = self
        gs = numpy.concatenate([me.gs, other.gs])
        ps = numpy.concatenate([me.ps, other.ps])
        cs = numpy.concatenate([me.cs, other.cs])
        return PauliPolynomial(gs, ps, cs=cs, qubits=me.qubits).reduce()

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # PauliPolynomial is iterable, numpy will broadcast ufunc calls. 
        # To prevent this unexpected behavior, intercept the call.
        for i, P in enumerate(inputs):
            if isinstance(P, PauliPolynomial):
                break
        if ufunc == numpy.multiply:
            return P.__rmul__(inputs[1-i])
        elif ufunc == numpy.add:
            return P.__add__(inputs[1-i])
        elif ufunc == numpy.subtract:
            if i == 0:
                return P.__add__(-inputs[1])
            else:
                return (-P).__add__(inputs[0])
        elif ufunc == numpy.divide:
            return P.__truediv__(inputs[1])
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, (Pauli, PauliMonomial, PauliPolynomial)):
            other = other.as_polynomial()
        else:
            raise NotImplementedError('matmul is not implemented for between {} and {}'.format(type(self).__name__, type(other).__name__))
        # If qubit sets differ, embed both to the union
        if not numpy.array_equal(self.qubits, other.qubits):
            qubits, inds1, inds2 = merge_qubits(self.qubits, other.qubits)
            # avoid in-place modification
            me = self.copy().embed_qubits(qubits, inds1)
            other = other.copy().embed_qubits(qubits, inds2)
        else:
            me = self
        gs, ps, cs = batch_dot(me.gs, me.ps, me.cs, other.gs, other.ps, other.cs)
        return PauliPolynomial(gs, ps, cs=cs, qubits=me.qubits).reduce()

    def trace(self):
        # normalized trace: 1 for identity operator, 0 otherwise
        return self.cs.dot(super(PauliPolynomial, self).trace())

    def inner(self, other):
        # normalized inner product: tr(self^dag other)/tr(I).
        if isinstance(other, (Pauli, PauliMonomial, PauliPolynomial)):
            other = other.as_polynomial()
        else:
            raise NotImplementedError('inner is not implemented for between {} and {}'.format(type(self).__name__, type(other).__name__))
        # If qubit sets differ, embed both to the union
        if not numpy.array_equal(self.qubits, other.qubits):
            qubits, inds1, inds2 = merge_qubits(self.qubits, other.qubits)
            # avoid in-place modification
            me = self.copy().embed_qubits(qubits, inds1)
            other = other.copy().embed_qubits(qubits, inds2)
        else:
            me = self
        # Broadcast Pauli strings comparison to find matches
        matches = numpy.all(me.gs[:,None,:] == other.gs[None,:,:], axis=-1)
        # Compute inner product using coefficients (True/False -> 1/0)
        return numpy.conjugate(me.cs) @ matches @ other.cs

    def copy(self):
        return PauliPolynomial(self.gs.copy(), self.ps.copy(), cs=self.cs.copy(), qubits=self.qubits.copy())

    def as_list(self):
        '''cast the Pauli polynomial to a Pauli list of operators
           ignore the coefficients '''
        return PauliList(self.gs, self.ps, qubits=self.qubits)

    def as_polynomial(self):
        return self

    def reduce(self, tol=1.e-10):
        '''Reduce the Pauli polynomial by 
            1. combine simiilar terms,
            2. move phase factors to coefficients,
            3. drop terms that are too small (coefficient < tol).'''
        gs, inds = numpy.unique(self.gs, return_inverse=True, axis=0)
        cs = aggregate(self.cs * 1j**self.ps, inds, gs.shape[0])
        mask = (numpy.abs(cs) > tol)
        return PauliPolynomial(gs[mask], cs=cs[mask], qubits=self.qubits)

    def krylov_space(self, max_dim=None, tol=1e-10, return_mat=False):
        '''Compute the Krylov space generated by a Pauli polynomial H.
            K = span(H^n for n = 0,1,2,...)
        
        Parameters:
        max_dim: int - maximum dimension of the Krylov space.
        tol: float - numerical tolerance for detecting linear dependence.
        return_mat: bool - whether to return the matrix representation of H in the Krylov space.

        Returns:
        basis: list[PauliPolynomial] (dim,) - orthonormal basis {K_i}
            of the Krylov space. (K_0 = I, K_1 = normalized H, ...)
        mat: complex (dim, dim) - the matrix representation M of H:
            M_{ij} = <K_i, H K_j>, where <A,B> = tr(A^dag B).
        '''
        # Initialize with identity operator
        basis = [pauli_identity_like(self)]
        cols = []  # List to collect matrix columns
        # If max_dim not provided, use theoretical bound 4^N for Pauli operators
        if max_dim is None:
            max_dim = 4**self.N
        for dim in range(1, max_dim): # i = dim = len(basis)
            col = numpy.zeros(dim + 1, dtype=cplx_type)
            A = self @ basis[-1] # A_i = H @ K_{i-1}
            for j in range(dim):
                col[j] = basis[j].inner(A) # M_{j,i-1} = <K_j, A_i>
                A = A - col[j] * basis[j] # A_i = A_i - M_{j,i-1} K_j
            col[dim] = numpy.sqrt(A.inner(A)) # M_{i,i} = <A_i, A_i>
            cols.append(col)
            if col[dim] < tol: # if A_i linearly dependent (residual norm too small)
                break # no new basis to add
            basis.append(A / col[dim]) # K_i = A_i / M_{i,i}
        if return_mat:
            # construct matrix from collected columns
            dim = len(basis)
            mat = numpy.zeros((dim, dim), dtype=cplx_type)
            for i, col in enumerate(cols):
                l = min(i+2, dim)
                mat[:l, i] = col[:l]  # fill column i with collected values 
            return basis, mat
        else:
            return basis
        
    def inverse(self):
        #Compute the inverse of a Pauli polynomial using its Krylov space representation.
        basis, mat = self.krylov_space(return_mat=True)
        iden = numpy.array([b.trace() for b in basis], dtype=mat.dtype)
        try:
            inv_vec = numpy.linalg.solve(mat, iden)
        except numpy.linalg.LinAlgError as e:
            raise RuntimeError(f"{self} is singular or ill-conditioned, and cannot be inverted: {e}")
        return sum(c * K for c, K in zip(inv_vec, basis))
    
    def __pow__(self, n):
        # Raise the PauliPolynomial to the power n.
        # For integer n, uses divide-and-conquer for efficiency.
        # For real/complex n, uses the Krylov space and matrix powering. 
        if isinstance(n, (int, numpy.integer)) or (isinstance(n, float) and n.is_integer()):
            n = int(n)
            if n == 0:
                return pauli_identity_like(self)
            elif n == 1:
                return self
            elif n == -1:
                return self.inverse()
            elif n < 0:
                return (self.inverse()) ** (-n)
            else:
                # divide and conquer: P^n = (P^(n/2))^2 * (P if n odd else I)
                half = self ** (n // 2)
                result = half @ half
                if n % 2:
                    result = result @ self
                return result
        # handle real/complex exponents using Krylov space
        basis, mat = self.krylov_space(return_mat=True)
        iden = numpy.array([b.trace() for b in basis], dtype=mat.dtype)
        try:
            mat_n = expm(n * logm(mat))
            vec = mat_n @ iden
        except Exception as e:
            raise RuntimeError(f"Failed to compute matrix power for non-integer exponent: {e}")
        # Linear combination of Krylov basis
        return sum(c * K for c, K in zip(vec, basis))
    
    def sqrt(self):
        return self ** 0.5
    
    def exp(self):
        # use the Krylov space and matrix exponential
        basis, mat = self.krylov_space(return_mat=True)
        iden = numpy.array([b.trace() for b in basis], dtype=mat.dtype)
        try:
            mat_exp = expm(mat)
            vec = mat_exp @ iden
        except Exception as e: 
            raise RuntimeError(f"Failed to compute matrix exponential: {e}")
        return sum(c * K for c, K in zip(vec, basis))
    
    def log(self):
        # use the Krylov space and matrix logarithm
        basis, mat = self.krylov_space(return_mat=True)
        iden = numpy.array([b.trace() for b in basis], dtype=mat.dtype)
        try:
            mat_log = logm(mat)
            vec = mat_log @ iden
        except Exception as e:
            raise RuntimeError(f"Failed to compute matrix logarithm: {e}")
        return sum(c * K for c, K in zip(vec, basis))

    def append(self, other):
        return NotImplemented
    
    def to_numpy(self, qubits=None):
        """Convert Pauli polynomial to numpy array representation.
        Returns a (2^N, 2^N) array representing the sum of all terms."""
        # Get batch of Pauli matrices from parent class (shape: L x 2^N x 2^N)
        matrices = super().to_numpy(qubits)
        # Contract batch dimension with coefficients to get final matrix
        return numpy.tensordot(self.cs, matrices, axes=(0,0))

# ---- constructors ----
import re
def pauli(obj, qubits = None):
    qubits = as_qubits(qubits) # handle qubits argument
    if isinstance(obj, Pauli):
        if qubits is None: # if qubits not provided
            return obj # no need to embed
        else: # if qubits is provided
            if numpy.array_equal(obj.qubits, qubits): # if qubits are identical
                return obj # no need to embed
            # check if obj.qubits is a subset of qubits
            if numpy.all(numpy.isin(obj.qubits, qubits)): # if so, embed
                return obj.embed_qubits(numpy.sort(qubits)) # need to ensure qubits are sorted
            else: # if not, raise error
                raise ValueError(f"The base qubits {obj.qubits} are not a subset of target qubits {qubits}, qubit embedding is not possible.")
    elif isinstance(obj, dict):
        keys, values = [], []
        for k, v in obj.items():
            try:
                k = int(k)
            except:
                if v in [4,5,6,7]:
                    values.append(v)
                elif isinstance(v, str):
                    values += list(v)
                else:
                    raise ValueError(f"Invalid phase modifier {v} from key {k}.")
                continue
            keys.append(k)
            values.append(v)
        keys = numpy.array(keys, dtype=qbit_int_type)
        raw = pauli(values, keys) # make a raw version
        return pauli(raw, qubits) # perform qubit embedding if needed
    elif isinstance(obj, str):
        dict_pattern = re.compile(r'([IXYZ])(?:_?\{?(\d+)\}?|\((\d+)\))')
        matches = list(dict_pattern.finditer(obj))
        if matches:
            coefficient = obj[:matches[0].start()].replace('j','i').replace('*','').strip()
            parse = {m.group(2) or m.group(3): m.group(1) for m in matches}
            phase_indx = {
                    '':     4, '+':     4, '-':     5,
                    '1':    4, '+1':    4, '-1':    5,
                    '1.0':  4, '+1.0':  4, '-1.0':  5,
                    'i':    6, '+i':    6, '-i':    7,
                    '1i':   6, '+1i':   6, '-1i':   7,
                    '1.0i': 6, '+1.0i': 6, '-1.0i': 7
                    }
            parse['phase'] = phase_indx.get(coefficient, 4)
            return pauli(parse, qubits)
        list_pattern = re.compile(r'([IXYZ])')
        matches = list(list_pattern.finditer(obj))
        if matches:
            coefficient = obj[:matches[0].start()].replace('j','i').replace('*','').strip()
            parse = list(coefficient) + [m.group(1) for m in matches]
            return pauli(parse, qubits)
        raise ValueError(f"Unable to parse Pauli string: {obj}")
    elif isinstance(obj, (tuple, list, numpy.ndarray, type({}.values()))):
        g = numpy.zeros(2*len(obj), dtype=indc_int_type)
        h = 0
        p = 0
        for i, mu in enumerate(obj):
            if mu == 0 or mu == 'I':
                continue
            elif mu == 1 or mu == 'X':
                g[2*(i-h)] = 1
            elif mu == 2 or mu == 'Y':
                g[2*(i-h)] = 1
                g[2*(i-h)+1] = 1
            elif mu == 3 or mu == 'Z':
                g[2*(i-h)+1] = 1
            elif mu == 4 or mu == '+':
                p = 0
                h += 1
            elif mu == 5 or mu == '-':
                p = 2
                h += 1
            elif mu == 'i':
                p += 1
                h += 1
            elif mu == 6:
                p = 1
                h += 1
            elif mu == 7:
                p = 3
                h += 1
            else:
                h += 1
        if h > 0:
            g = g[:-2*h]
        return Pauli(g, p, qubits = qubits)
    else:
        raise TypeError('pauli(obj) recieves obj of type {}, which is not implemented.'.format(type(obj).__name__))

import types
def paulis(*objs, qubits = None):
    qubits = as_qubits(qubits) # handle qubits argument
    # short cut if objs is empty
    if len(objs) == 0:
        N = 0 if qubits is None else len(qubits)
        gs = numpy.zeros((0,2*N), dtype=indc_int_type)
        return PauliList(gs, qubits=qubits)
    # short cut if PauliList is passed in
    if len(objs) == 1 :
        if isinstance(objs[0], PauliList):
            return objs[0]
        if isinstance(objs[0], (tuple, list, set, numpy.ndarray, types.GeneratorType)):
            objs = objs[0]
    # otherwise construct data for Pauli operators
    if qubits is None:
        # construct operator with qubits free (auto determined)
        ops = [pauli(obj) for obj in objs]
        # find common qubits
        qubits = numpy.unique(numpy.concatenate([op.qubits for op in ops])) 
        for op in ops: # embed operators
            op.embed_qubits(qubits) # in-place modification
    else: # construct operator with qubits requirements
        ops = [pauli(obj, qubits=qubits) for obj in objs]
    gs = numpy.stack([op.g for op in ops], dtype=indc_int_type)
    ps = numpy.array([op.p for op in ops], dtype=indc_int_type)
    return PauliList(gs, ps, qubits=qubits)

def pauli_identity(qubits = None):
    '''Pauli polynomial of an idenity operator of N qubits.'''
    qubits = as_qubits(qubits)
    if qubits is None:
        gs = numpy.zeros((1,0), dtype=indc_int_type)
    else:
        gs = numpy.zeros((1,2*len(qubits)), dtype=indc_int_type)
    return PauliPolynomial(gs, qubits=qubits)

def pauli_identity_like(other):
    '''Pauli polynomial of an identity operator of the same size as other.'''
    return pauli_identity(other.qubits)

def pauli_zero(qubits = None):
    '''Pauli polynomial of zero operator of N qubit'''
    return 0 * pauli_identity(qubits)

def pauli_zero_like(other):
    '''Pauli polynomial of zero operator of the same size as other.'''
    return pauli_zero(other.qubits)

def random_paulis(samples, qubits = None, seed = None):
    '''Sample a list of random Pauli operators of N qubits.
    
    Parameters:
    samples: int - number of samples to draw.
    qubits: int or list of int - qubits to be acted on.
    seed: int - random seed.

    Returns:
    PauliList - a list of random Pauli operators.
    '''
    qubits = as_qubits(qubits)
    N = 0 if qubits is None else len(qubits)
    if N > 0:
        if seed is not None:
            numpy.random.seed(seed)
        gs = numpy.random.randint(0,2,(samples,2*N), dtype=indc_int_type)
        ps = numpy.random.randint(0,4,samples, dtype=indc_int_type)
    else:
        gs = numpy.zeros((samples,0), dtype=indc_int_type)
        ps = numpy.zeros(samples, dtype=indc_int_type)
    return PauliList(gs, ps, qubits=qubits)