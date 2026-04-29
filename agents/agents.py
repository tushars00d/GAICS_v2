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

class NetworkAgent:
    """Proposes network-level containment strategies."""
    def analyze(self, anomaly, context):
        return {
            "proposed_action": "ISOLATE_SUBNET",
            "target": "vpc-subnet-1234",
            "rationale": "High-volume anomaly detected; isolating subnet prevents lateral movement.",
            "confidence": 0.85
        }

class IdentityAgent:
    """Proposes IAM-level containment strategies."""
    def analyze(self, anomaly, context):
        return {
            "proposed_action": "REVOKE_IAM_TOKEN",
            "target": "arn:aws:iam::123:role/CompromisedDev",
            "rationale": "Anomaly matches credential stuffing profiles; revoking session tokens stops API abuse.",
            "confidence": 0.92
        }

class SupervisorOrchestrator:
    """
    Evaluates conflicting proposals from Sub-Agents and determines the optimal 
    response based on Asset Criticality and Anomaly Severity.
    """
    def resolve_conflict(self, network_proposal, identity_proposal, asset_criticality):
        print("[Supervisor Orchestrator] Analyzing conflicting Agent proposals...")
        
        # Conflict Resolution Matrix Logic
        # If the asset is highly critical (e.g., Production DB), broad subnet isolation is too disruptive.
        # We favor surgical Identity revocation.
        if asset_criticality > 0.7:
            print("[Supervisor Orchestrator] Asset is HIGHLY CRITICAL. Rejecting disruptive Network Isolation.")
            print("[Supervisor Orchestrator] Selecting Surgical IAM Revocation.")
            return identity_proposal
        else:
            # For low criticality assets (e.g., Dev servers), nuke the subnet to be safe.
            print("[Supervisor Orchestrator] Asset is Low/Med Criticality. Favoring broad Network Isolation.")
            return network_proposal

class AutonomousResponseAgent:
    """
    Layer 4: Autonomous Response Engine.
    Integrates the RAG context (Layer 3), orchestrates sub-agents, 
    connects to Cloud APIs, and is gated by the Bayesian Governance Engine (Layer 5).
    """
    def __init__(self, config):
        self.config = config
        self.rag = CyberThreatRAG()
        self.cloud_api = MockCloudAPI()
        self.governance = TrustGovernanceEngine()
        
        self.network_agent = NetworkAgent()
        self.identity_agent = IdentityAgent()
        self.supervisor = SupervisorOrchestrator()
        
    def process_incident(self, telemetry_vector, severity_score=0.85, asset_criticality=0.9):
        print("\n--- Layer 3: Cognitive Analysis & Multi-Agent Orchestration ---")
        context = self.rag.retrieve_context(str(telemetry_vector))
        
        print("[*] Sub-Agents analyzing telemetry and context...")
        net_proposal = self.network_agent.analyze(telemetry_vector, context)
        id_proposal = self.identity_agent.analyze(telemetry_vector, context)
        
        print(f"   - Network Agent Proposal: {net_proposal['proposed_action']} (Conf: {net_proposal['confidence']})")
        print(f"   - Identity Agent Proposal: {id_proposal['proposed_action']} (Conf: {id_proposal['confidence']})")
        
        # Resolve Conflict
        final_playbook = self.supervisor.resolve_conflict(net_proposal, id_proposal, asset_criticality)
        print(f"[*] Final Approved Playbook: {json.dumps(final_playbook)}")
        
        print("\n--- Layer 5: Bayesian Governance Engine Validation ---")
        gov_result = self.governance.calculate_trusted_autonomy_score(
            llm_conf=final_playbook["confidence"],
            anomaly_sev=severity_score,
            asset_crit=asset_criticality
        )
        
        print(f"[*] PGM Evaluation: {gov_result['evidence']}")
        print(f"[*] Trusted Autonomy Score: {gov_result['trusted_autonomy_score']:.2f} -> {gov_result['decision']}")
        
        print("\n--- Layer 4: Autonomous Agentic Response ---")
        if gov_result["decision"] == "EXECUTE":
            api_result = self.cloud_api.execute_action(final_playbook["proposed_action"], final_playbook["target"])
            print(f"[*] Outcome: {api_result}")
        else:
            print("[*] Outcome: Action BLOCKED by Governance Engine. Routing to Human SOC Analyst.")
            
        return {
            "proposed_action": final_playbook,
            "governance_validation": gov_result,
            "status": gov_result["decision"]
        }
