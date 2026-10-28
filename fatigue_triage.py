from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Define the Network Structure
model = DiscreteBayesianNetwork([
    ('Sleep', 'Physical_Readiness'),
    ('Hydration', 'Physical_Readiness'),
    ('Physical_Readiness', 'Fatigue_Triage'),
    ('Mental_Load', 'Fatigue_Triage')
])

# 2. Define Individual Probabilities (Priors)
cpd_sleep = TabularCPD(variable='Sleep', variable_card=3, values=[[0.2], [0.5], [0.3]])
cpd_hydration = TabularCPD(variable='Hydration', variable_card=2, values=[[0.4], [0.6]])
cpd_mental = TabularCPD(variable='Mental_Load', variable_card=2, values=[[0.7], [0.3]])

# 3. Define Conditional Probabilities
cpd_physical = TabularCPD(
    variable='Physical_Readiness', variable_card=2,
    values=[
        [0.9, 0.8, 0.6, 0.4, 0.2, 0.1], 
        [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]  
    ],
    evidence=['Sleep', 'Hydration'], evidence_card=[3, 2]
)

cpd_fatigue = TabularCPD(
    variable='Fatigue_Triage', variable_card=5,
    values=[
        # Columns: 
        # [Weak Phys+Light Load, Weak Phys+Heavy Load, Strong Phys+Light Load, Strong Phys+Heavy Load]
        [0.05, 0.00, 0.80, 0.30], # Peak (Maximized only when Strong & Light Load)
        [0.20, 0.05, 0.15, 0.40], # Capable (Most likely when Strong but Heavy Load)
        [0.40, 0.20, 0.05, 0.20], # Strained
        [0.25, 0.45, 0.00, 0.10], # Depleted
        [0.10, 0.30, 0.00, 0.00]  # Critical (Spikes when Weak & Heavy Load)
    ],
    evidence=['Physical_Readiness', 'Mental_Load'], evidence_card=[2, 2],
    state_names={
        'Fatigue_Triage': ['Peak', 'Capable', 'Strained', 'Depleted', 'Critical'],
        'Physical_Readiness': [0, 1],
        'Mental_Load': [0, 1]
    }
)

# 4. Add CPTs to the Model and Validate
model.add_cpds(cpd_sleep, cpd_hydration, cpd_mental, cpd_physical, cpd_fatigue)
assert model.check_model() 

# 5. Exact State Input Validation
def get_valid_state(prompt, valid_options):
    while True:
        try:
            value = int(input(prompt))
            if value in valid_options:
                return value
            else:
                print(f">> Error: You must enter one of the following numbers: {valid_options}")
        except ValueError:
            print(">> Error: Invalid character. Please enter a number.")

# 6. Interactive Execution
if __name__ == "__main__":
    print("\n" + "="*45)
    print(" COGNITIVE FATIGUE TRIAGE INFERENCE ENGINE ")
    print("="*45)
    
    sleep_in = get_valid_state("Enter Sleep (0: <5 hrs, 1: 5-7 hrs, 2: >7 hrs): ", [0, 1, 2])
    hydro_in = get_valid_state("Enter Hydration (0: Low, 1: Optimal): ", [0, 1])
    mental_in = get_valid_state("Enter Mental Load (0: Light, 1: Heavy): ", [0, 1])

    # 7. Run Inference
    infer = VariableElimination(model)
    result = infer.query(
        variables=['Fatigue_Triage'], 
        evidence={'Sleep': sleep_in, 'Hydration': hydro_in, 'Mental_Load': mental_in}
    )
    
    print("\n" + "-"*40)
    print(" INFERRED FATIGUE PROBABILITIES ")
    print("-" *40)
    print(result)