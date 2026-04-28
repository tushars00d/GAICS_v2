import mlflow
import json
from datetime import datetime
import os

def setup_mlflow(config):
    experiment_name = config.get("evaluation", {}).get("mlflow_experiment_name", "GAICS_Dissertation")
    mlflow.set_experiment(experiment_name)

def log_agent_action(incident_data, decision):
    """
    Layer 5: Governance and action trace logging.
    Logs autonomous decisions into a structured JSON file.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "incident_features": incident_data,
        "mitre_mapping": decision.get("mitre_mapping", ""),
        "reasoning": decision.get("reasoning", ""),
        "action_taken": decision.get("action", "none"),
        "confidence": decision.get("confidence", 0.0)
    }
    
    os.makedirs("logs", exist_ok=True)
    with open("logs/agent_actions.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    print(f"[Governance Log] Action '{log_entry['action_taken']}' logged securely.")
