Cognitive Fatigue Triage Inference Engine

An Artificial Intelligence mini-project that utilizes a Discrete Bayesian Network to deduce the hidden probability of cognitive exhaustion based on observable daily metrics. Built for an academic AI coursework requirement, this engine handles real-world uncertainty (unlike strict propositional logic) to prescribe an actionable cognitive triage state.

🧠 System Architecture

The engine uses Probabilistic Reasoning to evaluate physiological and psychological inputs. The variables are modeled as a Directed Acyclic Graph (DAG) with intermediate hidden nodes to optimize the Conditional Probability Tables (CPTs).

Evidence Nodes: Sleep, Hydration, Mental Load.

Hidden Nodes: Physical Readiness.

Target Node (Fatigue Triage): Outputs a 5-level probability distribution:

Peak (Cleared for complex architecture/algorithms)

Capable (Baseline state for standard work)

Strained (Requires low-stakes tasks)

Depleted (Brain fog present, requires active recovery)

Critical (Complete exhaustion, mandates rest)

⚙️ Installation

This project requires Python 3 and the pgmpy library for probabilistic graphical models.

Clone the repository:

git clone https://github.com/yourusername/cognitive-fatigue-triage.git
cd cognitive-fatigue-triage


Create a virtual environment (optional but recommended):

python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate


Install the required dependency:

pip install pgmpy


🚀 Usage

Run the main script via the terminal. The engine features exact state input validation, ensuring it only accepts the mapped categorical indices to prevent mathematical errors.

python fatigue_triage.py


Sample Output

=============================================
 COGNITIVE FATIGUE TRIAGE INFERENCE ENGINE 
=============================================
Enter Sleep (0: <5 hrs, 1: 5-7 hrs, 2: >7 hrs): 0
Enter Hydration (0: Low, 1: Optimal): 0
Enter Mental Load (0: Light, 1: Heavy): 1

---------------------------------------------
 INFERRED FATIGUE PROBABILITIES 
---------------------------------------------
+--------------------------+---------------------+
| Fatigue_Triage           |   phi(Fatigue_Triage) |
+==========================+=====================+
| Fatigue_Triage(Peak)     |              0.0300 |
+--------------------------+---------------------+
| Fatigue_Triage(Capable)  |              0.0850 |
+--------------------------+---------------------+
| Fatigue_Triage(Strained) |              0.2000 |
+--------------------------+---------------------+
| Fatigue_Triage(Depleted) |              0.4150 |
+--------------------------+---------------------+
| Fatigue_Triage(Critical) |              0.2700 |
+--------------------------+---------------------+


📝 Technologies & Concepts Used

Python 3

pgmpy (DiscreteBayesianNetwork, TabularCPD, VariableElimination)

Artificial Intelligence: Knowledge Representation, Probability Theory, Bayesian Inference, and Uncertainty.
