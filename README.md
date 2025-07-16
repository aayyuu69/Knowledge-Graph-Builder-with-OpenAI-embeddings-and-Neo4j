# Knowledge Graph Builder with OpenAI Embeddings and Neo4j

A professional Python implementation for building intelligent knowledge graphs using OpenAI's text-embedding-ada-002 model and Neo4j graph database. This project demonstrates advanced semantic relationship modeling, graph construction, and intelligent querying capabilities suitable for enterprise knowledge management systems.

## 🎯 Overview

This Knowledge Graph Builder:
- Constructs semantic knowledge graphs from textual data
- Generates high-quality embeddings using OpenAI's text-embedding-ada-002 model
- Stores knowledge nodes and relationships in Neo4j graph database
- Enables intelligent querying through semantic similarity matching
- Provides scalable architecture for enterprise knowledge management

## 🏗️ Knowledge Graph Architecture

\`\`\`
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text Input    │───▶│  OpenAI API      │───▶│   Embeddings    │
│   (Knowledge)   │    │  (ada-002)       │    │   (1536-dim)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Knowledge       │◀───│  Semantic        │◀───│   Neo4j         │
│ Discovery       │    │  Relationships   │    │   Graph DB      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
\`\`\`

## 🚀 Setup Instructions

### Prerequisites
- Python 3.6+
- Neo4j Database (local, Docker, or Neo4j Aura)
- OpenAI API Key

### 1. Clone and Install Dependencies

\`\`\`bash
# Install required packages
pip install -r requirements.txt
\`\`\`

### 2. Set Up Neo4j Database

#### Option A: Local Neo4j Installation
1. Download and install Neo4j Desktop from [neo4j.com](https://neo4j.com/download/)
2. Create a new database with username \`neo4j\` and password \`password\`
3. Start the database (default URI: \`bolt://localhost:7687\`)

#### Option B: Docker
\`\`\`bash
docker run \\
    --name neo4j \\
    -p7474:7474 -p7687:7687 \\
    -d \\
    -e NEO4J_AUTH=neo4j/password \\
    neo4j:latest
\`\`\`

#### Option C: Neo4j Aura (Cloud)
1. Sign up at [neo4j.com/aura](https://neo4j.com/aura)
2. Create a free instance
3. Note the connection URI and credentials

### 3. Configure Environment Variables

Create a \`.env\` file in the project root:

\`\`\`env
OPENAI_API_KEY=your_openai_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
\`\`\`

### 4. Run the Application

\`\`\`bash
python scripts/semantic_search.py
\`\`\`

## 🔧 How the Knowledge Graph Works

### 1. Knowledge Ingestion
The system processes textual knowledge and converts it into semantic embeddings using OpenAI's `text-embedding-ada-002` model, creating 1536-dimensional vectors that capture deep semantic meaning.

### 2. Graph Construction
Knowledge nodes are stored in Neo4j with rich metadata and semantic embeddings:
\`\`\`cypher
(:KnowledgeNode {
    text: "Knowledge content",
    embedding: [1536-dimensional semantic vector],
    metadata: "Contextual information",
    created_at: datetime(),
    knowledge_type: "concept|fact|procedure"
})
\`\`\`

### 3. Semantic Relationship Discovery
- Query knowledge is converted to semantic embeddings
- Cosine similarity reveals hidden relationships between concepts
- Results ranked by semantic relevance (0-1 similarity score)

### 4. Intelligent Knowledge Retrieval
The system enables discovery of related knowledge through semantic understanding rather than keyword matching.

## 📊 Knowledge Graph Example

### Knowledge Query
\`\`\`
🔍 Knowledge Query: "machine learning algorithms for pattern recognition"
\`\`\`

### Knowledge Discovery Results
\`\`\`
🧠 KNOWLEDGE GRAPH RESULTS for: 'machine learning algorithms for pattern recognition'
================================================================================

1. Semantic Relevance: 0.8742
   Knowledge: Deep learning uses neural networks with multiple layers to model complex patterns in data.
   Type: AI/ML Concept
   Node ID: 123

2. Semantic Relevance: 0.7891
   Knowledge: Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.
   Type: AI/ML Definition  
   Node ID: 124

3. Semantic Relevance: 0.6543
   Knowledge: Natural language processing enables computers to understand and generate human language.
   Type: AI/ML Application
   Node ID: 125
\`\`\`

## 🗄️ Knowledge Graph Structure

The Neo4j knowledge graph uses an extensible semantic structure:

\`\`\`cypher
// Core knowledge nodes
(:KnowledgeNode)
├── text: String (knowledge content)
├── embedding: List<Float> (semantic vector)
├── metadata: String (contextual information)
├── knowledge_type: String (concept|fact|procedure|example)
├── domain: String (subject domain)
└── created_at: DateTime (ingestion timestamp)

// Semantic relationships (future extensions)
(:KnowledgeNode)-[:SEMANTICALLY_RELATED]->(:KnowledgeNode)
(:KnowledgeNode)-[:PART_OF]->(:Domain)
(:KnowledgeNode)-[:SUPPORTS]->(:Concept)
(:KnowledgeNode)-[:CONTRADICTS]->(:KnowledgeNode)
\`\`\`

## 🔍 Running Similarity Queries

### Building Knowledge Graphs Programmatically
\`\`\`python
from semantic_search import KnowledgeGraphBuilder

# Initialize knowledge graph builder
builder = KnowledgeGraphBuilder(uri, user, password, api_key)

# Add knowledge to the graph
node_id = builder.create_text_node(
    "Machine learning enables pattern recognition", 
    {"domain": "AI", "type": "concept"}
)

# Discover related knowledge
related_knowledge = builder.find_similar_texts("pattern recognition", top_k=5)

# Build knowledge relationships
for knowledge in related_knowledge:
    print(f"Relevance: {knowledge['similarity_score']:.4f}")
    print(f"Knowledge: {knowledge['text']}")
\`\`\`

### Interactive Mode
Run the script and follow the prompts for an interactive search experience.

## 🧪 Testing

The application includes sample data for immediate testing:
- Machine learning concepts
- Programming languages
- AI/ML topics

This allows you to test similarity search without adding your own data first.

## 🔧 Configuration Options

### Environment Variables
- \`OPENAI_API_KEY\`: Your OpenAI API key (required)
- \`NEO4J_URI\`: Neo4j connection URI (default: bolt://localhost:7687)
- \`NEO4J_USER\`: Neo4j username (default: neo4j)
- \`NEO4J_PASSWORD\`: Neo4j password (default: password)

### Customization
- Modify \`embedding_model\` in the class to use different OpenAI models
- Adjust \`top_k\` parameter in similarity search
- Add custom metadata fields for enhanced categorization

## 🚨 Error Handling

The application includes comprehensive error handling for:
- OpenAI API failures
- Neo4j connection issues
- Invalid input validation
- Network timeouts

## 📈 Performance Considerations

- Embeddings are cached in the database to avoid regeneration
- Cosine similarity calculations are optimized using NumPy
- Database queries use efficient Cypher patterns
- Consider indexing for large datasets

## 🔮 Knowledge Graph Enhancements

- **Automatic Relationship Discovery**: Build semantic relationships between knowledge nodes
- **Domain-Specific Ontologies**: Create specialized knowledge domains
- **Knowledge Validation**: Implement consistency checking and conflict resolution
- **Multi-Modal Knowledge**: Support for images, documents, and structured data
- **Knowledge Evolution Tracking**: Version control for evolving knowledge
- **Collaborative Knowledge Building**: Multi-user knowledge contribution workflows
- **Knowledge Recommendation Engine**: Suggest related knowledge for exploration

## 📝 License

This project is created for educational and evaluation purposes. Feel free to use and modify as needed.

---

**Project**: Knowledge Graph Builder  
**Author**: Vaibhav 

**Technologies**: Python, OpenAI API, Neo4j, Semantic AI, Graph Databases
