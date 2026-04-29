import yaml
import traceback
from agents.agents import AutonomousResponseAgent

try:
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    agent = AutonomousResponseAgent(config)
    decision = agent.process_incident(
        telemetry_vector=[2.5, -1.2, 0.9], 
        severity_score=0.92, 
        asset_criticality=0.85 
    )
    print("\n[SUCCESS] Phase 5 executed perfectly!")
except Exception as e:
    print("\n[ERROR] Phase 5 failed:")
    traceback.print_exc()
