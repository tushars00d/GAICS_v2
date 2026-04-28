import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class CyberThreatRAG:
    """
    Layer 3: RAG pipeline mapping incidents to MITRE ATT&CK.
    Uses FAISS vector store locally.
    """
    def __init__(self):
        # Using a small, local open-source embedding model for demonstration
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """
        Populate the vector store with dummy MITRE ATT&CK knowledge base.
        In production, this would parse official MITRE STIX data.
        """
        mitre_docs = [
            Document(page_content="T1078 - Valid Accounts. Adversaries may obtain and abuse credentials of existing accounts as a means of gaining Initial Access, Persistence, Privilege Escalation, or Defense Evasion.", metadata={"technique_id": "T1078"}),
            Document(page_content="T1110 - Brute Force. Adversaries may use brute force techniques to attempt access to accounts when passwords are unknown or when password hashes are obtained.", metadata={"technique_id": "T1110"}),
            Document(page_content="T1499 - Endpoint Denial of Service. Adversaries may perform Endpoint Denial of Service (DoS) attacks to degrade or block the availability of services to legitimate users.", metadata={"technique_id": "T1499"}),
            Document(page_content="T1046 - Network Service Discovery. Adversaries may attempt to get a listing of services listening on remote hosts and local network infrastructure devices.", metadata={"technique_id": "T1046"}),
            Document(page_content="T1059 - Command and Scripting Interpreter. Adversaries may abuse command and script interpreters to execute commands, scripts, or binaries.", metadata={"technique_id": "T1059"})
        ]
        
        self.vector_store = FAISS.from_documents(mitre_docs, self.embeddings)

    def retrieve_context(self, incident_description, k=2):
        """
        Retrieve top-k relevant MITRE ATT&CK techniques.
        """
        if not self.vector_store:
            return ""
            
        docs = self.vector_store.similarity_search(incident_description, k=k)
        context = "\n".join([f"[{doc.metadata['technique_id']}] {doc.page_content}" for doc in docs])
        return context
