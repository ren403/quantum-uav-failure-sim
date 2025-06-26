from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.circuit.library import UCCSD
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.settings import settings

settings.dict_aux_operators = True

# 🔁 Bond lengths simulating thermal cycling
bond_lengths = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]  # in Ångström
results = {}

for dist in bond_lengths:
    # Step 1: Define molecule
    molecule = f"H 0 0 0; H 0 0 {dist}"
    driver = PySCFDriver(atom=molecule, basis="sto3g")
    problem = ElectronicStructureProblem(driver)
    second_q_ops = problem.second_q_ops()
    main_op = second_q_ops[0]

    # Step 2: Qubit mapping
    converter = QubitConverter(JordanWignerMapper())
    qubit_op = converter.convert(main_op, problem.num_particles)

    # Step 3: VQE setup
    ansatz = UCCSD(qubit_converter=converter,
                   num_particles=problem.num_particles,
                   num_spatial_orbitals=problem.num_spatial_orbitals)
    optimizer = SLSQP(maxiter=1000)
    vqe = VQE(ansatz=ansatz, optimizer=optimizer, estimator=Estimator())

    # Step 4: Ground state solver
    solver = GroundStateEigensolver(converter, vqe)
    result = solver.solve(problem)

    energy = result.total_energies[0].real
    results[dist] = energy

    print(f"Thermal Cycle (Bond Length = {dist:.2f} Å): Energy = {energy:.6f} Hartree")
