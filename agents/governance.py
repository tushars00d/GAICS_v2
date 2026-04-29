import pgmpy.models as models
import pgmpy.factors.discrete as discrete
import numpy as np

class TrustGovernanceEngine:
    """
    Layer 5: Bayesian Governance Engine.
    Implements a Probabilistic Graphical Model (PGM) to mathematically validate 
    LLM decisions before execution, preventing hallucinated or overly destructive actions.
    """
    def __init__(self):
        # Define the structure of the Bayesian Network
        try:
            self.model = models.DiscreteBayesianNetwork([
                ('LLM_Confidence', 'Action_Approved'),
                ('Anomaly_Severity', 'Action_Approved'),
                ('Asset_Criticality', 'Action_Approved')
            ])
        except AttributeError:
            self.model = models.BayesianNetwork([
                ('LLM_Confidence', 'Action_Approved'),
                ('Anomaly_Severity', 'Action_Approved'),
                ('Asset_Criticality', 'Action_Approved')
            ])
        
        # Define Conditional Probability Tables (CPTs)
        
        # P(LLM_Confidence): 0 = Low, 1 = High
        cpd_conf = discrete.TabularCPD(variable='LLM_Confidence', variable_card=2, values=[[0.3], [0.7]])
        
        # P(Anomaly_Severity): 0 = Low, 1 = Critical
        cpd_sev = discrete.TabularCPD(variable='Anomaly_Severity', variable_card=2, values=[[0.8], [0.2]])
        
        # P(Asset_Criticality): 0 = Low (e.g. dev server), 1 = High (e.g. production DB)
        cpd_crit = discrete.TabularCPD(variable='Asset_Criticality', variable_card=2, values=[[0.9], [0.1]])
        
        # P(Action_Approved | LLM_Confidence, Anomaly_Severity, Asset_Criticality)
        # Shape: 2 x (2*2*2) = 2 x 8
        # Logic: We only approve (1) if Confidence is High AND (Severity is High OR Criticality is Low)
        # If Criticality is High, we are extremely conservative.
        
        # States: Conf(0/1), Sev(0/1), Crit(0/1)
        # Columns correspond to: (000, 001, 010, 011, 100, 101, 110, 111)
        # Probabilities of [Reject (0), Approve (1)]
        
        # We manually tune these priors based on risk appetite.
        approve_probs = [
            0.01, # 000: Low Conf, Low Sev, Low Crit -> Reject (Not needed)
            0.00, # 001: Low Conf, Low Sev, High Crit -> Reject
            0.05, # 010: Low Conf, High Sev, Low Crit -> Reject (Too risky)
            0.00, # 011: Low Conf, High Sev, High Crit -> Reject
            0.80, # 100: High Conf, Low Sev, Low Crit -> Approve
            0.10, # 101: High Conf, Low Sev, High Crit -> Reject (Overkill)
            0.95, # 110: High Conf, High Sev, Low Crit -> Approve (Slam dunk)
            0.40  # 111: High Conf, High Sev, High Crit -> 40% Approve (Needs Human in Loop mostly)
        ]
        
        reject_probs = [1 - p for p in approve_probs]
        
        cpd_app = discrete.TabularCPD(
            variable='Action_Approved', variable_card=2, 
            values=[reject_probs, approve_probs],
            evidence=['LLM_Confidence', 'Anomaly_Severity', 'Asset_Criticality'],
            evidence_card=[2, 2, 2]
        )
        
        self.model.add_cpds(cpd_conf, cpd_sev, cpd_crit, cpd_app)
        assert self.model.check_model()
        
        from pgmpy.inference import VariableElimination
        self.inference = VariableElimination(self.model)

    def calculate_trusted_autonomy_score(self, llm_conf: float, anomaly_sev: float, asset_crit: float) -> dict:
        """
        Maps continuous 0.0-1.0 scores to discrete bins and queries the PGM.
        """
        # Discretize inputs
        conf_state = 1 if llm_conf > 0.8 else 0
        sev_state = 1 if anomaly_sev > 0.7 else 0
        crit_state = 1 if asset_crit > 0.8 else 0
        
        # Query Bayesian Network
        result = self.inference.query(
            variables=['Action_Approved'], 
            evidence={
                'LLM_Confidence': conf_state,
                'Anomaly_Severity': sev_state,
                'Asset_Criticality': crit_state
            }
        )
        
        prob_approve = result.values[1]
        
        decision = "EXECUTE" if prob_approve >= 0.5 else "BLOCK_REQUEST_HUMAN"
        
        return {
            "trusted_autonomy_score": float(prob_approve),
            "decision": decision,
            "evidence": f"Conf={conf_state}, Sev={sev_state}, Crit={crit_state}"
        }
