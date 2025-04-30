# Midterm Progress Report: Adaptive Teacher-Student Learning Framework for Dynamic Load-Carrying in Bipedal Robots

## 1. Updated Project Definition and Goals

### Original Goals
- Phase 1: Reproduce quadrupedal robot results from Chang et al. [21]
- Phase 2: Extend framework to bipedal robots
- Optional: Enhance student model performance

### Current Status and Adjustments
- We have shifted focus directly to bipedal implementation (Phase 2) due to:
  - Availability of bipedal robot simulation environment (Isaac Gym)
  - More relevant codebase for our specific use case
  - Direct applicability to our research interests

### Modified Goals
1. Implement dynamic load-carrying capabilities in bipedal robot
2. Develop teacher-student learning framework
3. Evaluate performance on various terrains and load conditions

## 2. Current Member Roles and Collaboration Strategy

### Member Roles
- [Your Name]: Lead developer for robot simulation and RL implementation
- [Other Members]: [Their roles]

### Collaboration Strategy
- Code Management:
  - Using Git for version control
  - Main repository: [Repository URL]
  - Branch strategy: Feature branches for new implementations
- Communication:
  - Regular meetings: [Frequency and platform]
  - Documentation: Shared Google Docs for design decisions
  - Task tracking: [Tool used]

## 3. Proposed Approach

### Implementation Plan
1. **Environment Setup** (Completed)
   - Using Isaac Gym for simulation
   - Modified PointFoot robot with load-carrying capabilities
   - Implemented container structure for dynamic loads

2. **Teacher Network Implementation** (In Progress)
   ```python
   # PPO Implementation Structure
   class TeacherNetwork(nn.Module):
       def __init__(self):
           # Network architecture
           self.actor = ActorNetwork()
           self.critic = CriticNetwork()
           
       def forward(self, obs):
           # Forward pass implementation
           return action_distribution, value
   ```

3. **Training Pipeline** (Planned)
   - Reward function design
   - PPO hyperparameter tuning
   - Curriculum learning implementation

4. **Evaluation Framework** (Planned)
   - Performance metrics
   - Terrain generation
   - Load variation scenarios

## 4. Data/Simulator/Physical Platform

### Simulation Environment
- Platform: Isaac Gym
- Robot: PointFoot bipedal robot
- Control Level: Joint-level control
- Interface: Python API

### Evaluation Setup
1. **Environment Interface**
   ```python
   class PointFootWithLoad(PointFoot):
       def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
           # Environment initialization
           # Load and container setup
           # Observation space definition
   ```

2. **Control Abstraction**
   - Joint-level control for precise movement
   - State observation including load dynamics
   - Reward computation for stability and performance

## 5. Initial Results

### Completed Steps
1. **Environment Setup**
   - Successfully implemented PointFootWithLoad class
   - Added container structure for dynamic loads
   - Implemented state tracking and reward functions

2. **Current Implementation**
   - Load and container state tracking
   - Observation space definition
   - Basic reward structure
   - Environment reset functionality

### Code Example
```python
# Current implementation of load and container tracking
def get_observations(self):
    # Base observations
    obs = super().get_observations()
    
    # Add load and container states
    load_obs = torch.cat((
        self.load_relative_position * self.obs_scales.lin_vel,
        self.load_relative_velocity * self.obs_scales.lin_vel
    ), dim=-1)
    
    if self.has_container:
        container_obs = torch.cat((
            self.container_relative_position * self.obs_scales.lin_vel,
            self.container_relative_velocity * self.obs_scales.lin_vel
        ), dim=-1)
        obs = torch.cat((obs, container_obs), dim=-1)
    
    return obs
```

## 6. Current Reservations and Questions

### Technical Challenges
1. **Stability Issues**
   - Bipedal robots inherently less stable than quadrupeds
   - Need to carefully tune reward function for balance

2. **Load Dynamics**
   - Complex interaction between robot and dynamic load
   - Need to ensure proper physical simulation

### Open Questions
1. How to best implement the teacher-student transfer?
2. What metrics should we use to evaluate stability?
3. How to handle different load configurations?

## References
[21] L. Chang, Y. Nai, H. Chen, and L. Yang, "Beyond robustness: Learning unknown dynamic load adaptation for quadruped locomotion on rough terrain," arXiv preprint arXiv:2403.08211, 2024.

