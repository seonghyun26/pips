from tqdm import tqdm
from openmm import app
from policies.Alanine import NNPolicy
from potentials.md_utils import ForceReporter
from openmmtools.integrators import VVVRIntegrator
from plotting.PlotPathsAlanine import PlotPathsAlanine

import torch
import numpy as np
import mdtraj as md
import openmm as mm
import openmm.unit as unit


# Classes

class AlanineOpenMM():
    def __init__(self, start_file, device, overide=True, index=0):
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
        
        if overide:
            self.simulation.saveCheckpoint(f"./simulation_save_{self.index}")
            

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
        """
        Helper function to get the current positions of the atoms.
        """
        positions = self.simulation.context.getState(True).getPositions(True).value_in_unit(unit.nanometer)
        positions = torch.asarray(positions, dtype=torch.float, device=device)

        return positions


    def reset(self):
        """
        Resets the atoms to the initial positions and removes the current bias potential. 
        The minimizeEnergy() call cancels out the velocities. 
        """
        self.simulation.loadCheckpoint(f"./results/notebook/simulation_save_{self.index}")
        for i in range(len(self.pdb.positions)):
            self.external_force.setParticleParameters(i, i, [0, 0, 0])
        self.external_force.updateParametersInContext(self.simulation.context)

      
# Functions
  
def get_starting_positions(MDs):
    initial_positions = []
    for MD in MDs:
        initial_positions.append(MD.get_current_positions())

    initial_positions = torch.stack(initial_positions).view(len(MDs), -1)

    return initial_positions

def run_simulations(MDs, control):
    positions = []
    for idx, MD in enumerate(MDs):
        positions.append(MD.perform_OpenMM_step(control[idx].view(-1, 3)))

    positions = torch.stack(positions).view(len(MDs), -1)

    return positions

def reset(MDs):
    for MD in MDs:
        MD.reset()

def phi(x, end):
    x = x[:, -1, :].view(x.shape[0], -1, 3)

    px = torch.cdist(x, x)
    pend = torch.cdist(end, end).repeat(x.shape[0], 1, 1)

    t = (px - pend) ** 2
    cost_distance = torch.mean(t, dim=(1, 2))
    cost_distance_final = (cost_distance).exp() - 1.

    return cost_distance_final * 100

def control_cost(control, noise, R):
    uR = torch.einsum('bij, jj -> bij', control, R)
    uRu = torch.einsum('bij, bij -> bi', uR, control)
    uReps = torch.einsum('bij, bij -> bi', uR, noise)

    control_cost = (uRu/2 + uReps/2).sum(1)

    return control_cost


# Step 1: Configs
date = "0501-232453"
device = 'cuda:7'
T = 500.  # Time horizon
n_rollouts = 10 #15000 # Number of epochs
n_samples = 16 # Number of trajectories for each epoch
std = 0.1
lr = 0.0001

u_dim = 22 * 3 # Doesn't include the positions
dim = u_dim * 2 # Full dimension of the statespace

dt = torch.tensor(1.)
n_steps = int(T / dt)


# Step 2: Initialize the different OpenMM implemenations
MDs = []
for i in tqdm(range(0, n_samples)):
    pot = AlanineOpenMM('./potentials/files/AD_c7eq.pdb', device, index=i)
    MDs.append(pot)
    
# Step 3: Set loss functions
endMD = AlanineOpenMM('./potentials/files/AD_c7ax.pdb', device, overide=False)
ending_positions = endMD.get_closest_minima()
R = torch.eye(u_dim).to(device)

# Step 4: Policy
policy = NNPolicy(device, dims = u_dim, force=True, T=T, vel_given=False)
paths = torch.zeros((n_samples, n_steps+1, u_dim), device=device)
history_control = torch.zeros((n_samples, n_steps, u_dim), device=device)
history_noise = torch.zeros((n_samples, n_steps, u_dim), device=device)
path_cost = torch.zeros((n_samples), device=device)
std = torch.eye(u_dim).to(device) * (std)
eps_distr = torch.distributions.MultivariateNormal(torch.zeros(n_samples, u_dim).to(device), std)
lambda_ = R @ ((std ** 2) / dt)
lambda_ = lambda_[0, 0] 
plot = True
policy_optimizers = torch.optim.SGD(policy.parameters(), lr=lr)
    
# Step 6: Sampling
policy = torch.load(f"./results/poly/{date}/path_policy", map_location=device)
policy.vel_given = False

reset(MDs)
paths[:, 0, :] = get_starting_positions(MDs)
new_state = paths[:, 0, :].to(device)

for s in tqdm(range(0, n_steps)):
    x = new_state.detach()
    # t = torch.tensor(s * dt)
    t = (s * dt).clone().detach()

    # Sample control and noise
    update_u = policy(x, t)
    update_action = update_u * dt

    # Perform OpenMM step
    positions = run_simulations(MDs, update_action)
    new_state = positions.to(device)

    # Bookkeeping
    paths[:, s+1, :]  = new_state

# Reset the MD simulations
reset(MDs)

# Print results
phi_cost = phi(paths, ending_positions)
order = phi_cost.argsort()
colors = PlotPathsAlanine(paths, ending_positions)
            
# Save last set of trajectories for plotting
last_order = order
last_paths = paths.clone()

# Step 7: Plotting
STEP = 1
best = last_order[0]
best_traj = last_paths[best].view(last_paths[best].shape[0], -1, 3)

trajs = None
for j in range(0, int(best_traj.shape[0]), STEP):
    traj = md.load_pdb('./potentials/files/AD_c7ax.pdb')
    
    atoms = []
    traj.xyz = np.array(best_traj[j].cpu())
    
    if j == 0:
        trajs = traj
    else:
        trajs = trajs.join(traj)
