import os
import torch
import pytz
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from policies.Alanine import NNPolicy

from openmm import app
import openmm as mm
import openmm.unit as unit
from openmmtools.integrators import VVVRIntegrator

from potentials.md_utils import ForceReporter

class AlanineOpenMM():
    def __init__(self, start_file, device='cpu', overide=True, index=0):
        """
        Intialize the OpenMM configuration
        """
        self.index = index
        self.start_file = start_file
        self.device = device

        # Setup openMM
        self.forcefield = app.ForceField('amber99sbildn.xml', 'tip3p.xml')
        self.pdb = app.PDBFile(self.start_file)
        self.system = self.forcefield.createSystem(
            self.pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005
        )

        # Implement external force
        self.external_force = mm.CustomExternalForce("k*(fx*x + fy*y + fz*z)")

        # creating the parameters
        self.external_force.addGlobalParameter("k", 1000)
        self.external_force.addPerParticleParameter("fx")
        self.external_force.addPerParticleParameter("fy")
        self.external_force.addPerParticleParameter("fz")
        self.system.addForce(self.external_force)
        for i in range(len(self.pdb.positions)):
            self.external_force.addParticle(i, [0, 0, 0])

        # Setup integrator
        self.integrator = VVVRIntegrator(
            300 * unit.kelvin,  # temp
            1.0 / unit.picoseconds,  # collision rate
            1.0 * unit.femtoseconds)  # timestep
        self.integrator.setConstraintTolerance(0.00001)

        # Environment setup
        self.platform = mm.Platform.getPlatformByName('CUDA')
        self.properties = {'DeviceIndex': '0', 'Precision': 'mixed'}

        # Finalize simulation
        self.simulation = app.Simulation(self.pdb.topology, self.system, self.integrator,
                                    self.platform, self.properties)
        self.simulation.context.setPositions(self.pdb.positions)
        
        self.simulation.step(1)
        self.simulation.minimizeEnergy()
            
    def get_closest_minima(self):
        """
        Given the current state of the system (atom locations) the closes local minima in the Potential Energy surface is returned
        """
        self.simulation.step(1)
        self.simulation.minimizeEnergy()
        self.simulation.step(1)

        return self.get_current_positions()

    def perform_OpenMM_step(self, control=None):
        """
        Performs the OpenMM step based on the given control. This is achieved by setting the placeholder values. 
        """
        if control is not None:
            for i in range(control.shape[0]):
                self.external_force.setParticleParameters(i, i, control[i])
            self.external_force.updateParametersInContext(self.simulation.context)

        self.simulation.step(1)

        return self.get_current_positions()
    
    def get_current_positions(self):
        positions = self.simulation.context.getState(True).getPositions(True).value_in_unit(unit.nanometer)
        positions = torch.asarray(positions, dtype=torch.float, device=self.device)
        return positions
    
    def reset(self):
        for i in range(len(self.pdb.positions)):
            self.external_force.setParticleParameters(i, i, [0, 0, 0])
        self.external_force.updateParametersInContext(self.simulation.context)

class AlaninePotential():
    def __init__(self):
        super().__init__()
        self.open_file()

    def open_file(self):
        file = "./plotting/final_frame.dat"

        with open(file) as f:
            lines = f.readlines()

        dims = [90, 90]

        self.locations = torch.zeros((int(dims[0]), int(dims[1]), 2))
        self.data = torch.zeros((int(dims[0]), int(dims[1])))

        i = 0
        for line in lines[1:]:

            splits = line[0:-1].split(" ")
            vals = [y for y in splits if y != '']

            x = float(vals[0])
            y = float(vals[1])
            val = float(vals[-1])

            self.locations[i // 90, i % 90, :] = torch.tensor([x, y])
            self.data[i // 90, i % 90] = (val)  # / 503.)
            i = i + 1

    def potential(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp, loc.double(), p=2)
        index = distances.argmin(dim=1)

        x = torch.div(index, self.locations.shape[0], rounding_mode='trunc')  # index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        z = self.data[x, y]
        return z

    def drift(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp[:, :2].double(), loc.double(), p=2)
        index = distances.argsort(dim=1)[:, :3]

        x = index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        dims = torch.stack([x, y], 2)

        min = dims.argmin(dim=1)
        max = dims.argmax(dim=1)

        min_x = min[:, 0]
        min_y = min[:, 1]
        max_x = max[:, 0]
        max_y = max[:, 1]

        min_x_dim = dims[range(dims.shape[0]), min_x, :]
        min_y_dim = dims[range(dims.shape[0]), min_y, :]
        max_x_dim = dims[range(dims.shape[0]), max_x, :]
        max_y_dim = dims[range(dims.shape[0]), max_y, :]

        min_x_val = self.data[min_x_dim[:, 0], min_x_dim[:, 1]]
        min_y_val = self.data[min_y_dim[:, 0], min_y_dim[:, 1]]
        max_x_val = self.data[max_x_dim[:, 0], max_x_dim[:, 1]]
        max_y_val = self.data[max_y_dim[:, 0], max_y_dim[:, 1]]

        grad = -1 * torch.stack([max_y_val - min_y_val, max_x_val - min_x_val], dim=1)

        return grad

def get_starting_positions(MDs):
    initial_positions = []
    for MD in MDs:
        initial_positions.append(MD.get_current_positions())

    initial_positions = torch.stack(initial_positions).view(len(MDs), -1)

    return initial_positions

def new_dihedral(p):
    """http://stackoverflow.com/q/20305272/1128289"""
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array(
        [v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.arctan2(y, x)



# Plot basic config
plt.clf()
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
plt.xlim([-np.pi, np.pi])
plt.ylim([-np.pi, np.pi])
z_max = 32
circle_size = 1200
saddle_size = 2400


# Plot basic potential by contour
potential = AlaninePotential()
xs = np.arange(-np.pi, np.pi + .1, .1)
ys = np.arange(-np.pi, np.pi + .1, .1)
x, y = np.meshgrid(xs, ys)
inp = torch.tensor(np.array([x, y])).view(2, -1).T
z = potential.potential(inp)
z = z.view(y.shape[0], y.shape[1])
plt.contourf(xs, ys, z, levels=z_max, zorder=0)



# Plot start and target positions
angle_1 = [6, 8, 14, 16]
angle_2 = [1, 6, 8, 14]

startMD = AlanineOpenMM('./potentials/files/AD_c7eq.pdb', index=0)
startMD.reset()
start_positions = startMD.get_current_positions()
psi_start = [new_dihedral(start_positions[angle_1, :])]
phi_start = [new_dihedral(start_positions[angle_2, :])]
ax.scatter(phi_start, psi_start, edgecolors='black', c='w', zorder=z_max, s=circle_size)
# print("Start positions:")
# print(phi_start)
# print(psi_start)

endMD = AlanineOpenMM('./potentials/files/AD_c7ax.pdb', overide=False)
target = endMD.get_closest_minima()
psis_target = [new_dihedral(target[angle_1, :])]
phis_target = [new_dihedral(target[angle_2, :])]
ax.scatter(phis_target, psis_target, edgecolors='black', c='w', zorder=z_max, s=circle_size)
# print("Target positions:")
# print(phis_target)
# print(psis_target)


phis_saddle = [-0.035, -0.017]
psis_saddle = [1.605, -0.535]
ax.scatter(phis_saddle, psis_saddle, edgecolors='black', c='w', zorder=z_max, s=saddle_size, marker="*")



# Plot config 2
plt.xlabel('\u03A6', fontsize=24, fontweight='medium')
plt.ylabel('\u03A8', fontsize=24, fontweight='medium')
plt.tick_params(
    left = False,
    right = False ,
    labelleft = False , 
    labelbottom = False,
    bottom = False
) 
plt.tight_layout()
plt.show()
plt.savefig("alanine.pdf", dpi=300)
print("File saved!")

