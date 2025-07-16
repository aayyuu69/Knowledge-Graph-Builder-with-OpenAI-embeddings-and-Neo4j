"""
Knowledge Graph Builder with OpenAI Embeddings and Neo4j
A professional implementation for building semantic knowledge graphs with intelligent querying.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from neo4j import GraphDatabase
import openai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class KnowledgeGraphBuilder:
    """
    A knowledge graph builder that uses OpenAI embeddings and Neo4j graph database
    to create semantic relationships and enable intelligent querying of knowledge.
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str):
        """
        Initialize the knowledge graph builder.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            openai_api_key: OpenAI API key
        """
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Embedding model configuration
        self.embedding_model = "text-embedding-ada-002"
        self.embedding_dimension = 1536  # Dimension for ada-002 model
        
        print("✅ Knowledge Graph Builder initialized successfully!")
    
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        print("🔌 Knowledge graph connection closed.")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate semantic embedding for knowledge content using OpenAI's text-embedding-ada-002.
        
        Args:
            text: Knowledge content to generate embedding for
            
        Returns:
            List of floats representing the semantic embedding vector
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            print(f"🧠 Generated semantic embedding for: '{text[:50]}...'")
            return embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            raise
    
    def add_knowledge_node(self, knowledge_text: str, metadata: Optional[Dict] = None) -> str:
        """
        Add a knowledge node to the graph with semantic embedding.
        
        Args:
            knowledge_text: The knowledge content
            metadata: Optional metadata (domain, type, source, etc.)
            
        Returns:
            Node ID of the created knowledge node
        """
        # Generate semantic embedding for the knowledge
        embedding = self.generate_embedding(knowledge_text)
        
        # Prepare metadata with defaults
        if metadata is None:
            metadata = {}
        
        # Set default knowledge type if not specified
        if 'knowledge_type' not in metadata:
            metadata['knowledge_type'] = 'concept'
        
        # Create knowledge node in Neo4j
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (k:KnowledgeNode {
                    text: $text,
                    embedding: $embedding,
                    metadata: $metadata,
                    knowledge_type: $knowledge_type,
                    domain: $domain,
                    created_at: datetime()
                })
                RETURN id(k) as node_id
                """,
                text=knowledge_text,
                embedding=embedding,
                metadata=json.dumps(metadata),
                knowledge_type=metadata.get('knowledge_type', 'concept'),
                domain=metadata.get('domain', 'general')
            )
            node_id = result.single()["node_id"]
            print(f"📚 Added knowledge node with ID: {node_id}")
            return str(node_id)
    
    def get_all_knowledge_nodes(self) -> List[Dict]:
        """
        Retrieve all knowledge nodes from the graph.
        
        Returns:
            List of dictionaries containing knowledge node information
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (k:KnowledgeNode)
                RETURN id(k) as node_id, k.text as text, k.embedding as embedding, 
                       k.metadata as metadata, k.knowledge_type as knowledge_type,
                       k.domain as domain, k.created_at as created_at
                ORDER BY k.created_at
                """
            )
            nodes = []
            for record in result:
                nodes.append({
                    "node_id": record["node_id"],
                    "text": record["text"],
                    "embedding": record["embedding"],
                    "metadata": json.loads(record["metadata"]) if record["metadata"] else {},
                    "knowledge_type": record["knowledge_type"],
                    "domain": record["domain"],
                    "created_at": record["created_at"]
                })
            return nodes
    
    def semantic_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate semantic similarity between two knowledge embeddings.
        
        Args:
            vec1: First knowledge embedding vector
            vec2: Second knowledge embedding vector
            
        Returns:
            Semantic similarity score (0-1, where 1 is most semantically similar)
        """
        # Convert to numpy arrays for efficient computation
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity for semantic relationship
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        return float(similarity)
    
    def discover_related_knowledge(self, query_text: str, top_k: int = 5, domain_filter: Optional[str] = None) -> List[Dict]:
        """
        Discover knowledge semantically related to the query.
        
        Args:
            query_text: Knowledge query to search for related content
            top_k: Number of top related knowledge items to return
            domain_filter: Optional domain filter (e.g., 'AI', 'programming')
            
        Returns:
            List of dictionaries with semantic relevance scores and knowledge information
        """
        # Generate semantic embedding for query
        query_embedding = self.generate_embedding(query_text)
        
        # Get knowledge nodes (with optional domain filtering)
        all_nodes = self.get_all_knowledge_nodes()
        
        if domain_filter:
            all_nodes = [node for node in all_nodes if node.get('domain', '').lower() == domain_filter.lower()]
        
        if not all_nodes:
            print("⚠️ No knowledge nodes found in the graph.")
            return []
        
        # Calculate semantic similarities
        related_knowledge = []
        for node in all_nodes:
            relevance_score = self.semantic_similarity(query_embedding, node["embedding"])
            related_knowledge.append({
                "node_id": node["node_id"],
                "text": node["text"],
                "semantic_relevance": relevance_score,
                "knowledge_type": node["knowledge_type"],
                "domain": node["domain"],
                "metadata": node["metadata"],
                "created_at": node["created_at"]
            })
        
        # Sort by semantic relevance (descending)
        related_knowledge.sort(key=lambda x: x["semantic_relevance"], reverse=True)
        
        # Return top k most relevant knowledge
        return related_knowledge[:top_k]
    
    def clear_knowledge_graph(self):
        """Clear all knowledge nodes from the graph."""
        with self.driver.session() as session:
            result = session.run("MATCH (k:KnowledgeNode) DELETE k RETURN count(k) as deleted_count")
            deleted_count = result.single()["deleted_count"]
            print(f"🗑️ Cleared {deleted_count} knowledge nodes from graph.")

def setup_sample_knowledge(builder: KnowledgeGraphBuilder):
    """
    Set up sample knowledge for demonstrating the knowledge graph builder.
    
    Args:
        builder: KnowledgeGraphBuilder instance
    """
    sample_knowledge = [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming.",
            "metadata": {"domain": "AI", "knowledge_type": "definition", "source": "educational"}
        },
        {
            "text": "Python is a high-level, interpreted programming language known for its simplicity, readability, and extensive ecosystem of libraries.",
            "metadata": {"domain": "programming", "knowledge_type": "definition", "source": "educational"}
        },
        {
            "text": "Deep learning utilizes artificial neural networks with multiple layers to model and understand complex patterns in large datasets.",
            "metadata": {"domain": "AI", "knowledge_type": "concept", "source": "educational"}
        },
        {
            "text": "JavaScript is a versatile, dynamic programming language primarily used for web development, enabling interactive user experiences.",
            "metadata": {"domain": "programming", "knowledge_type": "definition", "source": "educational"}
        },
        {
            "text": "Natural language processing combines computational linguistics with machine learning to enable computers to understand human language.",
            "metadata": {"domain": "AI", "knowledge_type": "concept", "source": "educational"}
        },
        {
            "text": "Graph databases like Neo4j excel at storing and querying highly connected data with complex relationships between entities.",
            "metadata": {"domain": "database", "knowledge_type": "concept", "source": "educational"}
        }
    ]
    
    print("🌱 Building sample knowledge graph...")
    for item in sample_knowledge:
        builder.add_knowledge_node(item["text"], item["metadata"])
    print("✅ Sample knowledge graph construction complete!")

def display_knowledge_nodes(nodes: List[Dict]):
    """Display knowledge nodes in a formatted way."""
    print("\n" + "="*80)
    print("📚 KNOWLEDGE GRAPH NODES")
    print("="*80)
    
    for i, node in enumerate(nodes, 1):
        print(f"\n{i}. Node ID: {node['node_id']}")
        print(f"   Knowledge: {node['text']}")
        print(f"   Type: {node['knowledge_type']}")
        print(f"   Domain: {node['domain']}")
        print(f"   Metadata: {node['metadata']}")
        print(f"   Created: {node['created_at']}")

def display_knowledge_discovery(results: List[Dict], query: str):
    """Display knowledge discovery results in a formatted way."""
    print("\n" + "="*80)
    print(f"🧠 KNOWLEDGE DISCOVERY RESULTS for: '{query}'")
    print("="*80)
    
    if not results:
        print("No related knowledge found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Semantic Relevance: {result['semantic_relevance']:.4f}")
        print(f"   Knowledge: {result['text']}")
        print(f"   Type: {result['knowledge_type']}")
        print(f"   Domain: {result['domain']}")
        print(f"   Node ID: {result['node_id']}")

def main():
    """
    Main function to demonstrate the Knowledge Graph Builder.
    """
    print("🚀 Starting Knowledge Graph Builder Demo")
    print("="*50)
    
    # Configuration (you can also use environment variables)
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable is required!")
        return
    
    try:
        # Initialize the knowledge graph builder
        builder = KnowledgeGraphBuilder(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Clear existing knowledge (optional - comment out to preserve existing knowledge)
        builder.clear_knowledge_graph()
        
        # Build sample knowledge graph
        setup_sample_knowledge(builder)
        
        # Display all knowledge nodes
        all_nodes = builder.get_all_knowledge_nodes()
        display_knowledge_nodes(all_nodes)
        
        # Interactive knowledge discovery
        print("\n" + "="*80)
        print("🎯 INTERACTIVE KNOWLEDGE DISCOVERY")
        print("="*80)
        print("Enter queries to discover related knowledge (or 'quit' to exit):")
        
        while True:
            user_input = input("\n🔍 Knowledge Query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                print("Please enter a valid knowledge query.")
                continue
            
            # Discover related knowledge
            related_results = builder.discover_related_knowledge(user_input, top_k=3)
            display_knowledge_discovery(related_results, user_input)
        
        # Close the connection
        builder.close()
        print("\n👋 Thank you for using the Knowledge Graph Builder!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your Neo4j connection and OpenAI API key.")

if __name__ == "__main__":
    main()
