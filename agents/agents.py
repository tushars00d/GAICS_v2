import json
import os
from langchain_core.prompts import PromptTemplate
from agents.rag_pipeline import CyberThreatRAG
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class AutonomousResponseAgent:
    """
    Layer 3 & 4: Multi-Agent System + Autonomous Response (SOAR).
    Comprises a Decomposer Agent and a Supervisor Agent.
    """
    def __init__(self, config):
        self.config = config
        self.rag = CyberThreatRAG()
        
        provider = config.get("layer3_agents", {}).get("llm_provider", "local")
        model_name = config.get("layer3_agents", {}).get("model_name", "gpt-3.5-turbo")
        temperature = config.get("layer3_agents", {}).get("temperature", 0.0)
        
        if provider == "openai" and "OPENAI_API_KEY" in os.environ:
            self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        else:
            # Fallback to dummy LLM for execution without API keys
            self.llm = DummyLLM()

    def process_incident(self, incident_data):
        """
        Execute the Decomposer -> Supervisor pipeline.
        """
        # Step 1: Decomposer Agent generates search query
        incident_desc = f"Anomalous traffic detected. Features: {json.dumps(incident_data)}"
        
        # Step 2: RAG Retrieval
        context = self.rag.retrieve_context(incident_desc)
        
        # Step 3: Supervisor Agent Decision
        system_prompt = """
        You are an autonomous SOAR Supervisor Agent.
        Analyze the incident data and the provided MITRE ATT&CK context.
        Output ONLY a JSON object with the following schema:
        {
            "mitre_mapping": "TXXXX",
            "reasoning": "brief explanation",
            "action": "isolate_host|block_ip|revoke_credentials|none",
            "confidence": 0.0-1.0
        }
        """
        
        user_prompt = f"Context:\n{context}\n\nIncident:\n{incident_desc}"
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # Parse JSON
        try:
            decision = json.loads(response.content)
        except:
            # Fallback for parsing errors
            decision = {
                "mitre_mapping": "Unknown",
                "reasoning": "Failed to parse LLM output",
                "action": "none",
                "confidence": 0.0
            }
            
        # Step 4: Autonomous Response Execution
        self._execute_action(decision)
        return decision

    def _execute_action(self, decision):
        """
        Layer 4: Deterministic SOAR simulation.
        """
        action = decision.get("action", "none")
        confidence = decision.get("confidence", 0.0)
        threshold = self.config.get("layer4_response", {}).get("confidence_threshold", 0.85)
        
        if confidence < threshold:
            print(f"[SOAR] Confidence {confidence} below threshold {threshold}. Action requires human approval.")
            return
            
        if not self.config.get("layer4_response", {}).get("simulate_soar", True):
            print(f"[SOAR] Simulation disabled. Proposed action: {action}")
            return

        if action == "isolate_host":
            print("[SOAR EXECUTED] Host isolated from the network via NAC API.")
        elif action == "block_ip":
            print("[SOAR EXECUTED] Malicious IP blocked at the perimeter firewall.")
        elif action == "revoke_credentials":
            print("[SOAR EXECUTED] User session terminated and credentials revoked.")
        else:
            print(f"[SOAR] No action taken. Evaluated state: {action}")


class DummyLLM:
    """Mock LLM for testing without API keys."""
    def invoke(self, messages):
        class DummyResponse:
            def __init__(self):
                self.content = '{"mitre_mapping": "T1499", "reasoning": "High volume traffic detected indicative of DoS", "action": "block_ip", "confidence": 0.95}'
        return DummyResponse()
