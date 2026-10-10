# 🔬 An LLM-Augmented Mechanism-Aware World Model for Coordinated Decision-Making in Industrial Production-Energy Systems

This repository provides the code implementation of the paper **"An LLM-Augmented Mechanism-Aware World Model for Coordinated Decision-Making in Industrial Production-Energy Systems"**.

## Project Introduction

This project focuses on coordinated modeling and decision-making for industrial production-energy systems.  
The proposed framework, **LMA-PEWM**, combines task-specific large language models, production-energy interaction modeling, mechanism-aware state restoration, and Actor-Critic behavior learning.

The experiment mainly includes the following modules:

- **Task-specific multi-LLM semantic state inference**
  - Production state inference
  - Energy state inference
  - Production-energy coupling state inference
- **Interaction-enhanced latent dynamics**
  - State encoding
  - Variable-Step GRU (VS-GRU)
  - Multi-Branch Cross Attention (MBCA)
  - Joint production-energy state fusion
- **LMA-PEWM world model learning**
  - Multi-step production-energy state prediction
  - Reward and feasibility prediction
  - Mechanism-aware regularization
- **KKT-Newton Local Mechanism Restoration (KKT-NLMR)**
  - Production-side mechanism projection
  - Energy-side mechanism projection
  - Local restoration of mechanism-violating imagined states
- **Actor-Critic behavior model learning**
  - Joint production-energy action generation
  - Imagined rollouts using the frozen world model
  - Long-term return optimization
- **Task-specific SFT data preparation**
  - Production
  - Energy
  - Production-energy coupling

## File Structure

```plaintext
LMA-PEWM/
│
├── patches/                                   # Patch-level production-energy data
│
├── sft_LLM/                                   # Task-specific LLM data preparation
│   │
│   ├── coupling/                              # Production-energy coupling task
│   │   ├── production_energy_coupled.py
│   │   ├── production_energy_coupled_messages_test.json
│   │   ├── production_energy_coupled_messages_test.jsonl
│   │   ├── production_energy_coupled_messages_train.json
│   │   └── production_energy_coupled_messages_train.jsonl
│   │
│   ├── energy/                                # Energy task
│   │   ├── energy_llm_train_dataset.json
│   │   ├── energy_llm_train_dataset.jsonl
│   │   └── steel_energy_plans.py
│   │
│   └── production/                            # Production task
│       ├── production_router_messages_test.jsonl
│       ├── production_router_messages_train.jsonl
│       └── steel_multi_plans.py
│
├── a_data_loader.py                           # Data loading and preprocessing
├── b_base_chat_model.py                       # Local fine-tuned LLM inference wrapper
│
├── c1_production_state_adapter.py             # Production state construction
├── c2_prod_agent_adapter.py                   # Production LLM semantic inference
│
├── d1_energy_state_adapter.py                 # Energy state construction
├── d2_energy_agent_adapter.py                 # Energy LLM semantic inference
│
├── e1_couple_state_assembler.py               # Production-energy state alignment and joint sample construction
├── e2_couple_agent_adapter.py                 # Coupling LLM semantic inference
│
├── f_transition_engine.py                     # Multi-step transition and rollout orchestration
├── g_lma_pewm.py                              # Core LMA-PEWM architecture
├── h_world_model_training.py                  # Stage-1 world-model training and validation
│
├── mechanism_energy_projection.py             # Energy-side mechanism-consistent projection / restoration
├── mechanism_schedule_projection.py           # Production-side mechanism-consistent projection / restoration
│
├── data_of_IES.xlsx                           # Industrial energy-system data
├── heat_segments.json                         # Heat-segment information
├── heat_segments.xlsx
├── patch_metadata.json                        # Patch metadata
├── patch_metadata.xlsx
├── production_daily_objects.json              # Daily production objects
│
├── main.py                                    # Stage-1 LMA-PEWM training entry point
├── main_RL.py                                 # Stage-2 Actor-Critic behavior learning entry point
│
├── README.md
└── requirements.txt
```

## Method Pipeline

The complete training procedure contains two sequential learning stages after task-specific LLM preparation.

```plaintext
Task-Specific LLM Preparation
        │
        ├── Production LLM
        ├── Energy LLM
        └── Production-Energy Coupling LLM
        │
        ▼
Stage 1: LMA-PEWM World Model Learning
        │
        ├── Semantic State Encoding
        ├── VS-GRU
        ├── MBCA
        ├── Joint State Fusion
        ├── Multi-Step Latent Dynamics
        └── Mechanism-Aware Training
        │
        ▼
Trained World Model: lma_pewm_best.pt
        │
        │  Freeze LMA-PEWM parameters
        ▼
Stage 2: Actor-Critic Behavior Model Learning
        │
        ├── Actor generates joint production-energy actions
        ├── Frozen LMA-PEWM generates imagined trajectories
        ├── KKT-NLMR repairs mechanism-violating states
        ├── Critic estimates state values
        └── Actor-Critic optimizes long-term return
        │
        ▼
Coordinated Production-Energy Decision Policy
```

## Environmental Requirements

The project is implemented in Python and mainly depends on the following packages:

- Python
- PyTorch
- Transformers
- NumPy
- Pandas
- OpenPyXL
- SciPy
- tqdm

Other dependencies can be installed using:

```bash
pip install -r requirements.txt
```

The task-specific LLMs are loaded from local fine-tuned model directories.  
Please modify the corresponding model paths according to your local environment before running the experiments.

## Run Steps

### 1. Prepare Task-Specific LLM Data

The `sft_LLM` directory contains scripts and datasets for production, energy, and production-energy coupling tasks.

#### Production task

```bash
python sft_LLM/production/steel_multi_plans.py
```

#### Energy task

```bash
python sft_LLM/energy/steel_energy_plans.py
```

#### Production-energy coupling task

```bash
python sft_LLM/coupling/production_energy_coupled.py
```

After preparing the datasets, fine-tune the corresponding task-specific LLMs and configure the local model paths used by the state-inference modules.

---

### 2. Stage 1: Train the LMA-PEWM World Model

The first learning stage trains the production-energy world model.

```bash
python main.py
```

The default pipeline loads the task-specific fine-tuned LLMs, generates semantic state records, and trains the LMA-PEWM.

For a small pipeline test:

```bash
python main.py --mode pipeline --days 2024-01-10 --history_window 5 --horizon 5 --stride 1 --max_samples_per_day 5 --epochs 2 --batch_size 2
```

For full Stage-1 training:

```bash
python main.py --mode pipeline --history_window 5 --horizon 5 --stride 1 --epochs 50 --batch_size 16
```

The main Stage-1 outputs are saved in:

```plaintext
world_model_artifacts/
│
├── semantic_world_model_records.jsonl
├── lma_pewm_best.pt
├── lma_pewm_last.pt
├── world_model_training_history.json
├── world_model_training_summary.json
└── stage1_training_config.json
```

The most important output is:

```plaintext
world_model_artifacts/lma_pewm_best.pt
```

This trained world model is required by Stage 2.

---

### 3. Stage 2: Train the Actor-Critic Behavior Model

Stage 2 is performed **after Stage 1 is completed**.

The trained LMA-PEWM is loaded and frozen. Only the Actor and Critic networks are updated during behavior-model learning.

```bash
python main_RL.py
```

For a small test:

```bash
python main_RL.py --epochs 2 --batch_size 4 --imagination_horizon 5
```

For full behavior-model training:

```bash
python main_RL.py --epochs 50 --batch_size 32 --imagination_horizon 15
```

The main Stage-2 outputs are saved in:

```plaintext
behavior_model_artifacts/
│
├── actor_critic_best.pt
├── actor_critic_last.pt
├── behavior_training_history.json
├── behavior_training_history.csv
└── behavior_training_summary.json
```

To evaluate a trained behavior model:

```bash
python main_RL.py --mode evaluate
```

## Notes

- Stage 1 and Stage 2 are trained **sequentially rather than jointly**.
- The three task-specific LLMs are used for semantic state inference in Stage 1.
- During Stage 2, the trained LMA-PEWM is frozen.
- KKT-NLMR is applied to mechanism-violating imagined states before subsequent state prediction and behavior-model updates.
- Large model weights and training checkpoints are not recommended for direct storage in the Git repository.
- Private industrial data should be removed or anonymized before making the repository public.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{LMA_PEWM,
  title   = {An LLM-Augmented Mechanism-Aware World Model for Coordinated Decision-Making in Industrial Production-Energy Systems},
  author  = {},
  journal = {},
  year    = {}
}
```

The complete citation information will be updated after publication.

## Contact Information

If there are any questions about the code or datasets, please feel free to contact us. Thanks!
