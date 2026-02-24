"""
📊 PDF CVs → Neo4j Knowledge Graph Pipeline
=============================================

Extracts text from PDF CVs and converts them into a knowledge graph
using LangChain's LLMGraphTransformer, then stores everything in Neo4j.

This creates the static knowledge base for the programmer-staffing
GraphRAG system.  The pipeline is:
    PDF  →  unstructured text  →  LLMGraphTransformer  →  Neo4j graph.

🔧 Prerequisites:
    - Azure OpenAI credentials in .env file
    - Neo4j running (see 0_setup.py / start_session.sh)
    - Generated CV PDFs (see 1_generate_data.py)

🎯 What You'll Learn:
    - How to extract structured knowledge from unstructured PDF documents
    - How LLMGraphTransformer maps text onto a predefined ontology
    - How to persist and validate a knowledge graph in Neo4j
"""

import asyncio
import logging
import textwrap
from pathlib import Path
from typing import Any

import toml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import AzureChatOpenAI
from neo4j.exceptions import GqlError
from unstructured.partition.pdf import partition_pdf

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_CV_NODE_TYPES: list[str] = [
    "Person", "Company", "University", "Skill", "Technology",
    "Project", "Certification", "Location", "JobTitle", "Industry",
]

ALLOWED_CV_RELATIONSHIP_TUPLES: list[tuple[str, str, str]] = [
    ("Person", "WORKED_AT", "Company"),
    ("Person", "STUDIED_AT", "University"),
    ("Person", "HAS_SKILL", "Skill"),
    ("Person", "LOCATED_IN", "Location"),
    ("Person", "HOLDS_POSITION", "JobTitle"),
    ("Person", "WORKED_ON", "Project"),
    ("Person", "EARNED", "Certification"),
    ("JobTitle", "AT_COMPANY", "Company"),
    ("Project", "USED_TECHNOLOGY", "Technology"),
    ("Project", "FOR_COMPANY", "Company"),
    ("Company", "IN_INDUSTRY", "Industry"),
    ("Skill", "RELATED_TO", "Technology"),
    ("Certification", "ISSUED_BY", "Company"),
    ("University", "LOCATED_IN", "Location"),
]

EXTRACTABLE_NODE_PROPERTIES: list[str] = [
    "start_date", "end_date", "level", "years_experience",
]

NEO4J_PERFORMANCE_INDEX_STATEMENTS: list[str] = [
    "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.id)",
    "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.id)",
    "CREATE INDEX skill_name IF NOT EXISTS FOR (s:Skill) ON (s.id)",
    "CREATE INDEX entity_base IF NOT EXISTS FOR (e:__Entity__) ON (e.id)",
]

GRAPH_VALIDATION_QUERIES: dict[str, str] = {
    "Total nodes": "MATCH (n) RETURN count(n) as count",
    "Total relationships": "MATCH ()-[r]->() RETURN count(r) as count",
    "Node types": "MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC",
    "Relationship types": "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC",
}

SAMPLE_RELATIONSHIP_QUERIES: list[str] = [
    "MATCH (p:Person)-[:HAS_SKILL]->(s:Skill) RETURN p.id, s.id LIMIT 5",
    "MATCH (p:Person)-[:WORKED_AT]->(c:Company) RETURN p.id, c.id LIMIT 5",
]


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


def load_toml_config(config_file_path: str = "utils/config.toml") -> dict[str, Any]:
    """Load and return the TOML configuration file as a dictionary."""
    resolved_path = Path(config_file_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    with resolved_path.open("r", encoding="utf-8") as config_handle:
        return toml.load(config_handle)


class CvKnowledgeGraphBuilder:
    """
    Builds a Neo4j knowledge graph from PDF CVs using LangChain's LLMGraphTransformer.

    The full pipeline executed by this builder:
      1. Connect to Neo4j and wipe all existing data for a fresh start
      2. Initialize an LLMGraphTransformer with a CV-specific ontology
      3. Extract text from each CV PDF via the *unstructured* library
      4. Convert extracted text into graph documents (nodes + relationships)
      5. Persist graph documents in Neo4j with performance indexes
      6. Validate the resulting graph by printing statistics and samples
    """

    def __init__(self, config_file_path: str = "utils/config.toml") -> None:
        self.config = load_toml_config(config_file_path)
        self.neo4j_graph = self._connect_and_reset_neo4j()
        self.cv_graph_transformer = self._build_cv_graph_transformer()

    def _connect_and_reset_neo4j(self) -> Neo4jGraph:
        """Connect to Neo4j, wipe all data, constraints, and indexes for a fresh start."""
        try:
            neo4j_connection = Neo4jGraph()
            logger.info("✓ Connected to Neo4j successfully")
        except (GqlError, ConnectionError, OSError) as connection_error:
            logger.error("Failed to connect to Neo4j: %s", connection_error)
            raise

        logger.info("Performing complete Neo4j cleanup…")
        self._perform_full_database_cleanup(neo4j_connection)
        logger.info("✓ Neo4j completely cleared")
        return neo4j_connection

    @staticmethod
    def _perform_full_database_cleanup(neo4j_connection: Neo4jGraph) -> None:
        """Delete every node/relationship, drop all constraints and indexes."""
        try:
            logger.info("  - Deleting all nodes and relationships…")
            neo4j_connection.query("MATCH (n) DETACH DELETE n")

            logger.info("  - Dropping all constraints…")
            for constraint_row in neo4j_connection.query("SHOW CONSTRAINTS"):
                constraint_name = constraint_row.get("name", "")
                if constraint_name:
                    try:
                        neo4j_connection.query(f"DROP CONSTRAINT {constraint_name}")
                    except GqlError as constraint_drop_error:
                        logger.debug("    Could not drop constraint %s: %s", constraint_name, constraint_drop_error)

            logger.info("  - Dropping all indexes…")
            for index_row in neo4j_connection.query("SHOW INDEXES"):
                index_name = index_row.get("name", "")
                if index_name and not index_name.startswith("__"):
                    try:
                        neo4j_connection.query(f"DROP INDEX {index_name}")
                    except GqlError as index_drop_error:
                        logger.debug("    Could not drop index %s: %s", index_name, index_drop_error)

            remaining_node_count = neo4j_connection.query("MATCH (n) RETURN count(n) as count")[0]["count"]
            remaining_rel_count = neo4j_connection.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]

            if remaining_node_count == 0 and remaining_rel_count == 0:
                logger.info("  ✓ Database completely clean")
            else:
                logger.warning(
                    "  ⚠ Cleanup incomplete: %d nodes, %d relationships remain",
                    remaining_node_count, remaining_rel_count,
                )

        except GqlError as cleanup_error:
            logger.error("Error during cleanup: %s", cleanup_error)
            logger.info("  - Falling back to basic cleanup…")
            neo4j_connection.query("MATCH (n) DETACH DELETE n")

    @staticmethod
    def _build_cv_graph_transformer() -> LLMGraphTransformer:
        """
        Create an LLMGraphTransformer configured with the CV-specific ontology.

        The transformer uses Azure OpenAI (gpt-4.1-mini) with temperature=0 for
        deterministic extraction.  Strict mode ensures only nodes and relationships
        from the predefined CV schema are produced.
        """
        azure_chat_llm = AzureChatOpenAI(model="gpt-4.1-mini", temperature=0)

        cv_transformer = LLMGraphTransformer(
            llm=azure_chat_llm,
            allowed_nodes=ALLOWED_CV_NODE_TYPES,
            allowed_relationships=ALLOWED_CV_RELATIONSHIP_TUPLES,
            node_properties=EXTRACTABLE_NODE_PROPERTIES,
            strict_mode=True,
        )

        logger.info("✓ LLM Graph Transformer initialized with CV schema")
        return cv_transformer

    @staticmethod
    def extract_text_from_pdf(pdf_file_path: str) -> str:
        """Extract full text content from a single PDF using the *unstructured* library."""
        try:
            parsed_elements = partition_pdf(filename=pdf_file_path)
            extracted_full_text = "\n\n".join(str(element) for element in parsed_elements)
            logger.debug("Extracted %d characters from %s", len(extracted_full_text), pdf_file_path)
            return extracted_full_text
        except (OSError, ValueError) as pdf_extraction_error:
            logger.error("Failed to extract text from %s: %s", pdf_file_path, pdf_extraction_error)
            return ""

    async def convert_single_cv_to_graph_documents(self, pdf_file_path: str) -> list:
        """
        Convert one CV PDF into LangChain graph documents via the LLM transformer.

        Steps: extract text → wrap as LangChain Document → asynchronously transform
        into nodes and relationships using the CV ontology.
        """
        logger.info("Processing: %s", Path(pdf_file_path).name)

        extracted_cv_text = self.extract_text_from_pdf(pdf_file_path)
        if not extracted_cv_text.strip():
            logger.warning("No text extracted from %s", pdf_file_path)
            return []

        cv_langchain_document = Document(
            page_content=extracted_cv_text,
            metadata={"source": pdf_file_path, "type": "cv"},
        )

        try:
            produced_graph_documents = await self.cv_graph_transformer.aconvert_to_graph_documents(
                [cv_langchain_document],
            )
            logger.info("✓ Extracted graph from %s", Path(pdf_file_path).name)

            if produced_graph_documents:
                extracted_nodes_count = len(produced_graph_documents[0].nodes)
                extracted_relationships_count = len(produced_graph_documents[0].relationships)
                logger.info("  - Nodes: %d, Relationships: %d", extracted_nodes_count, extracted_relationships_count)

            return produced_graph_documents

        except (ValueError, RuntimeError) as conversion_error:
            logger.error("Failed to convert %s to graph: %s", pdf_file_path, conversion_error)
            return []

    async def process_all_cv_pdfs(self, cv_directory_path: str | None = None) -> int:
        """Process every PDF in *cv_directory_path* and store the resulting graph in Neo4j."""
        if cv_directory_path is None:
            cv_directory_path = self.config["output"]["programmers_dir"]

        cv_pdf_directory = Path(cv_directory_path)
        sorted_pdf_files = sorted(cv_pdf_directory.glob("*.pdf"))

        if not sorted_pdf_files:
            logger.error("No PDF files found in %s", cv_directory_path)
            return 0

        logger.info("Found %d PDF files to process", len(sorted_pdf_files))

        successfully_processed_count = 0
        accumulated_graph_documents: list = []

        for pdf_file in sorted_pdf_files:
            single_cv_graph_documents = await self.convert_single_cv_to_graph_documents(str(pdf_file))
            if single_cv_graph_documents:
                accumulated_graph_documents.extend(single_cv_graph_documents)
                successfully_processed_count += 1
            else:
                logger.warning("Failed to process %s", pdf_file)

        if accumulated_graph_documents:
            logger.info("Storing graph documents in Neo4j…")
            self._store_graph_documents_in_neo4j(accumulated_graph_documents)

        return successfully_processed_count

    def _store_graph_documents_in_neo4j(self, graph_documents_to_store: list) -> None:
        """Persist graph documents in Neo4j and create performance indexes."""
        try:
            self.neo4j_graph.add_graph_documents(
                graph_documents_to_store,
                baseEntityLabel=True,
                include_source=True,
            )

            total_stored_nodes = sum(len(doc.nodes) for doc in graph_documents_to_store)
            total_stored_relationships = sum(len(doc.relationships) for doc in graph_documents_to_store)

            logger.info("✓ Stored %d documents in Neo4j", len(graph_documents_to_store))
            logger.info("✓ Total nodes: %d", total_stored_nodes)
            logger.info("✓ Total relationships: %d", total_stored_relationships)

            self._create_performance_indexes()

        except (GqlError, RuntimeError) as storage_error:
            logger.error("Failed to store graph documents: %s", storage_error)
            raise

    def _create_performance_indexes(self) -> None:
        """Create Neo4j indexes for faster lookups on frequently-queried node types."""
        for index_statement in NEO4J_PERFORMANCE_INDEX_STATEMENTS:
            try:
                self.neo4j_graph.query(index_statement)
                logger.debug("Created index: %s", index_statement)
            except GqlError as index_creation_error:
                logger.debug("Index might already exist: %s", index_creation_error)

    def validate_graph(self) -> None:
        """Print statistics and sample relationships to verify the graph was built correctly."""
        logger.info("Validating knowledge graph…")

        for query_description, cypher_query in GRAPH_VALIDATION_QUERIES.items():
            try:
                query_result = self.neo4j_graph.query(cypher_query)
                if query_description in ("Total nodes", "Total relationships"):
                    logger.info("%s: %s", query_description, query_result[0]["count"])
                else:
                    logger.info("\n%s:", query_description)
                    for result_row in query_result[:10]:
                        if "type" in result_row:
                            logger.info("  %s: %s", result_row["type"], result_row["count"])
                        else:
                            logger.info("  %s", result_row)
            except GqlError as validation_error:
                logger.error("Failed to execute validation query '%s': %s", query_description, validation_error)

        logger.info("\nSample relationships:")
        for sample_query in SAMPLE_RELATIONSHIP_QUERIES:
            try:
                for result_row in self.neo4j_graph.query(sample_query):
                    logger.info("  %s", dict(result_row))
            except GqlError as sample_query_error:
                logger.debug("Sample query failed: %s", sample_query_error)


async def main() -> None:
    """Convert all generated CV PDFs into a Neo4j knowledge graph."""
    print_section_header("Converting PDF CVs to Knowledge Graph")

    try:
        graph_builder = CvKnowledgeGraphBuilder()
        processed_cv_count = await graph_builder.process_all_cv_pdfs()

        if processed_cv_count > 0:
            graph_builder.validate_graph()
            print(textwrap.dedent(f"""\

                ✓ Successfully processed {processed_cv_count} CV(s)
                ✓ Knowledge graph created in Neo4j

                Next steps:
                1. Run: uv run python 3_query_knowledge_graph.py
                2. Open Neo4j Browser to explore the graph
                3. Try GraphRAG queries!"""))
        else:
            print(textwrap.dedent("""\
                ❌ No CVs were successfully processed
                Please check the PDF files in data/programmers/ directory"""))

    except FileNotFoundError as file_error:
        logger.error("File not found: %s", file_error)
        print(f"❌ Error: {file_error}")
    except (GqlError, ConnectionError) as database_error:
        logger.error("Database error: %s", database_error)
        print(f"❌ Database Error: {database_error}")
    except RuntimeError as runtime_error:
        logger.error("Failed to build knowledge graph: %s", runtime_error)
        print(f"❌ Error: {runtime_error}")


if __name__ == "__main__":
    asyncio.run(main())
