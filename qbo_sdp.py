import time
import cvxpy as cvx
import numpy as np
import operator
import json
import sys



def qbo_sdp_01(Q, linear_constraint=None, quadratic_constraints=None, 
               solver=cvx.MOSEK, verbose=False, **solver_kwargs):
    """Compute SDP relaxation of QBO problem x^TQx, x in {0, 1}^n

    Args:
        Q (square numpy 2d array): A matrix Q.
        linear_constraint (Tuple(A, b, operator), optional): A linear constraint. Defaults to None.
        quadratic_constraints (Iterable(Tuple(Qi, ri)), optional): A collection of quadratic constraints. Defaults to None.
        solver (cvx.solver, optional): A solver for the SDP relaxation. Defaults to cvx.MOSEK.
        verbose (bool, optional): Verbosity of cvx.solver. Defaults to False.

    Raises:
        TBD
    Returns:
        TBD
    """
    # Allowed (in)equality operators in constraints
    allowed_operators = {'<=': operator.le, '>=': operator.ge, '==': operator.eq}

    def check_operator(operator_as_string):
        """Check operator.

        Args:
            operator_as_string (string): A string representation of (in)equality operator - python syntax.

        Raises:
            ValueError: Raises ValueError of not allowed operator is used.

        Returns:
            TBD
        """
        try:
            return allowed_operators[operator_as_string]
        except KeyError:
            raise ValueError(f'Operator {operator_as_string} is not allowed. Please use one of {list(allowed_operators.keys())}')
        except Exception as ex:
            raise(ex)
        
    
    def check_linear_constraint(linear_constraint, n):
        """TBD

        Args:
            linear_constraint (_type_): _description_
            n (_type_): _description_

        Raises:
            ValueError: _description_
            ex: _description_
            ValueError: _description_
            ex: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        A, b, operator_as_string = linear_constraint
        _operator = check_operator(operator_as_string)
        try:
            m_A, n_A = A.shape
        except ValueError:
            raise ValueError(f'A has to be a numpy 2D array of a shape (m, {n}). Got: {A.shape}')
        except Exception as ex:
            raise ex()
            
        try:
            m_b, = b.shape
        except ValueError:
            raise ValueError(f'b has to be a numpy 1D array. Got: {b.shape}')
        except Exception as ex:
            raise ex()

        if not (n == n_A or m_A == m_b):
            raise ValueError(f'Incompatible dimensions. n={n}, A=({m_A}, {n_A}), b=({m_b})')

        return A, b, _operator

    def check_quadratic_constraints(quadratic_constrains, n):
        """TBD

        Args:
            quadratic_constrains (_type_): _description_
            n (_type_): _description_

        Raises:
            ValueError: _description_
            ex: _description_
            ValueError: _description_

        Yields:
            _type_: _description_
        """
        for i, (Qi, ri, operator_as_string) in enumerate(quadratic_constraints):
            _operator = check_operator(operator_as_string)
            try:
                m_Qi, n_Qi = Qi.shape
            except ValueError:
                raise ValueError(f'Q{i} has to be a square numpy 2D array of a shape ({n}, {n}). Got: {Qi.shape}')
            except Exception as ex:
                raise ex()   

            if not (m_Qi == n_Qi and n == m_Qi):
                raise ValueError(f'Q{i} has to be a square numpy 2D array of a shape ({n}, {n}). Got: {Qi.shape}')

            yield Qi, ri, _operator
            
    
    start_time = time.time()

    n, _ = Q.shape
    
    Y = cvx.Variable((n+1, n+1), symmetric=True)
    X = Y[1:, 1:]
    x = cvx.diag(X)
    constraints = [
        Y >> 0, 
        Y[0, 0]== 1, 
        Y[0, 1:] == x, 
        x >= 0, x <= 1
    ]

    objective = cvx.Minimize(cvx.trace(Q @ X))

    if linear_constraint is not None:
        A, b, _operator = check_linear_constraint(linear_constraint, n)
        A_m, _ = A.shape
        
        constraints += [
            _operator(A[i, :] @ x, b[i]) for i in range(A_m)
        ]


    if quadratic_constraints is not None:
        constraints += [
            _operator(cvx.trace(Qi @ X), ri) for Qi, ri, _operator in check_quadratic_constraints(quadratic_constraints, n)
        ]
        
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose, **solver_kwargs)
    runtime = time.time() - start_time

    
    return (problem.value,
            (x.value,
            X.value,
            problem.status,
            runtime))



if __name__ == "__main__":

    file = sys.argv[1]
    print('Computing problem instance', file)

    with open(file) as f:
        data = json.load(f)


    Q = np.asarray(data['QBO']['Q'])
    linear_constraint = data['QBO']['constraints']['linear']
    quadratic_constraints = data['QBO']['constraints']['quadratic']
    if linear_constraint is not None:
        A, b, _operator = linear_constraint
        linear_constraint = np.asarray(A), np.asarray(b), _operator

    if quadratic_constraints:
        quadratic_constraints = ((np.asarray(Qi), ri, _operator) for Qi, ri, _operator in quadratic_constraints)


    val, _ = qbo_sdp_01(Q, solver=cvx.CVXOPT, linear_constraint=linear_constraint, quadratic_constraints=quadratic_constraints, verbose=True)

    print('Computed lower bound: ', val)
    print('Optimal value:', data['optimum'])

