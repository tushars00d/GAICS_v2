import json
import os
import torch
from langchain.prompts import PromptTemplate
from agents.rag_pipeline import CyberThreatRAG
from agents.governance import TrustGovernanceEngine

class MockCloudAPI:
    """
    Simulates AWS Boto3 / Azure Management APIs to prove the execution bridge.
    """
    def execute_action(self, action_name, target):
        print(f"\n[CLOUD API] Executing SOAR Playbook: {action_name} on target: {target}")
        if action_name == "ISOLATE_SUBNET":
            return f"Success: Modified Security Group to deny all ingress for {target}."
        elif action_name == "REVOKE_IAM_TOKEN":
            return f"Success: Revoked active sessions for IAM role {target}."
        else:
            return f"Warning: Unrecognized action {action_name}."

class AutonomousResponseAgent:
    """
    Layer 4: Autonomous Response Engine.
    Integrates the RAG context (Layer 3), connects to Cloud APIs, 
    and is gated by the Bayesian Governance Engine (Layer 5).
    """
    def __init__(self, config):
        self.config = config
        self.rag = CyberThreatRAG()
        self.cloud_api = MockCloudAPI()
        self.governance = TrustGovernanceEngine()
        
        self.prompt = PromptTemplate(
            input_variables=["context", "anomaly"],
            template="""
            You are a Cloud Security Orchestrator Agent.
            Anomaly Detected: {anomaly}
            Threat Intelligence Context: {context}
            
            Determine the single best mitigation action from: [ISOLATE_SUBNET, REVOKE_IAM_TOKEN, PASS]
            Output ONLY a JSON block like: {{"action": "ACTION_NAME", "target": "target_id", "confidence": 0.95}}
            """
        )
        
    def process_incident(self, telemetry_vector, severity_score=0.85, asset_criticality=0.9):
        print("\n--- Layer 3: Cognitive Analysis ---")
        context = self.rag.retrieve_context(str(telemetry_vector))
        
        # Simulate an LLM decision based on the RAG context
        print("[LLM Agent] Reasoning over RAG Context...")
        
        # In a full deployment, you pass self.prompt to an actual LangChain LLM here.
        # For execution speed in this defense pipeline, we simulate the LLM's parsed JSON output:
        llm_response = {
            "action": "REVOKE_IAM_TOKEN", 
            "target": "arn:aws:iam::123:role/CompromisedDev", 
            "confidence": 0.92
        }
        
        print(f"[*] Proposed Playbook: {json.dumps(llm_response)}")
        
        print("\n--- Layer 5: Bayesian Governance Engine Validation ---")
        gov_result = self.governance.calculate_trusted_autonomy_score(
            llm_conf=llm_response["confidence"],
            anomaly_sev=severity_score,
            asset_crit=asset_criticality
        )
        
        print(f"[*] PGM Evaluation: {gov_result['evidence']}")
        print(f"[*] Trusted Autonomy Score: {gov_result['trusted_autonomy_score']:.2f} -> {gov_result['decision']}")
        
        print("\n--- Layer 4: Autonomous Agentic Response ---")
        if gov_result["decision"] == "EXECUTE":
            api_result = self.cloud_api.execute_action(llm_response["action"], llm_response["target"])
            print(f"[*] Outcome: {api_result}")
        else:
            print("[*] Outcome: Action BLOCKED by Governance Engine. Routing to Human SOC Analyst.")
            
        return {
            "proposed_action": llm_response,
            "governance_validation": gov_result,
            "status": gov_result["decision"]
        }
