from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import yaml
from agents.agents import AutonomousResponseAgent
from utils.logging_utils import log_agent_action

app = FastAPI(
    title="GAICS Autonomous Response API",
    description="Layer 4 API for ingesting telemetry and executing SOAR actions.",
    version="1.0.0"
)

# Load config and agent globally
with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)
agent = AutonomousResponseAgent(config)

class TelemetryInput(BaseModel):
    timestamp: str
    source_ip: str
    destination_ip: str
    features: List[float] # Continuous tabular features

class IncidentReport(BaseModel):
    mitre_technique: str
    reasoning: str
    action_taken: str
    confidence: float

@app.post("/ingest", response_model=IncidentReport)
def ingest_telemetry(payload: TelemetryInput):
    try:
        # Pass features to the Multi-Agent System
        decision = agent.process_incident(payload.features)
        
        # Log the action (Layer 5)
        log_agent_action(payload.features, decision)
        
        return IncidentReport(
            mitre_technique=decision.get("mitre_mapping", "Unknown"),
            reasoning=decision.get("reasoning", "None provided"),
            action_taken=decision.get("action", "none"),
            confidence=decision.get("confidence", 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
