"""
🎓 GraphRAG Educational Setup System
=====================================

Interactive setup that handles fresh installations and existing databases,
designed for educational use with clear explanations and safe operations.

Usage:
    python 0_setup.py            # Interactive mode (default)
    python 0_setup.py --init     # Initialize Neo4j and directories only
    python 0_setup.py --fresh    # Force fresh start
    python 0_setup.py --continue # Continue with existing data
    python 0_setup.py --check    # Just check status
    python 0_setup.py --learning # Educational mode with explanations
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from neo4j.exceptions import GqlError

load_dotenv(override=True)

NEO4J_BOLT_URL = "bolt://localhost:7687"
NEO4J_BROWSER_URL = "http://localhost:7474"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password123"
NEO4J_CONTAINER_NAME = "neo4j-graphrag"

DOCKER_COMPOSE_TEMPLATE = """\
version: '3.8'
services:
  neo4j:
    image: neo4j:latest
    container_name: neo4j-graphrag
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password123
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j-data:/data
      - neo4j-logs:/logs
      - neo4j-import:/var/lib/neo4j/import
      - neo4j-plugins:/plugins
    restart: unless-stopped

volumes:
  neo4j-data:
  neo4j-logs:
  neo4j-import:
  neo4j-plugins:
"""

PROJECT_DIRECTORIES = [
    "data/programmers",
    "data/projects",
    "data/RFP",
    "results",
    "utils",
]


def print_section_header(title: str) -> None:
    separator = "=" * 70
    print(f"\n{separator}\n{title}\n{separator}")


@dataclass(frozen=True)
class DockerNeo4jStatus:
    docker_available: bool = False
    neo4j_container_id: str | None = None
    container_running: bool = False


@dataclass(frozen=True)
class Neo4jConnectionResult:
    connected: bool = False
    uri: str = NEO4J_BOLT_URL
    version: str = "unknown"


@dataclass(frozen=True)
class DatabaseStats:
    total_nodes: int = 0
    total_relationships: int = 0
    node_types: dict[str, int] = field(default_factory=dict)
    programmer_count: int = 0
    error_message: str | None = None


@dataclass(frozen=True)
class DataIntegrityReport:
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SystemState:
    connection: Neo4jConnectionResult
    stats: DatabaseStats
    integrity: DataIntegrityReport
    has_data: bool
    has_programmers: bool


class SetupMode(Enum):
    INTERACTIVE = "interactive"
    FRESH = "fresh"
    CONTINUE = "continue"
    CHECK = "check"
    LEARNING = "learning"
    INIT = "init"


@dataclass(frozen=True)
class ParsedSetupConfig:
    mode: SetupMode
    learning_mode: bool


def detect_docker_neo4j_status() -> DockerNeo4jStatus:
    """Detect whether Docker is available and whether a Neo4j container exists."""
    try:
        docker_ps_result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.ID}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if docker_ps_result.returncode != 0:
            return DockerNeo4jStatus()

        for line in docker_ps_result.stdout.strip().split("\n"):
            if line and NEO4J_CONTAINER_NAME in line:
                parts = line.split("\t")
                if len(parts) >= 3:
                    return DockerNeo4jStatus(
                        docker_available=True,
                        neo4j_container_id=parts[1],
                        container_running="Up" in parts[2],
                    )

        return DockerNeo4jStatus(docker_available=True)

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return DockerNeo4jStatus()


class Neo4jStatusChecker:
    """Manages the connection to a Neo4j instance and inspects its state."""

    def __init__(self) -> None:
        self._graph: Neo4jGraph | None = None

    def check_connection(self) -> Neo4jConnectionResult:
        """Attempt to connect to the Neo4j database and retrieve its version."""
        try:
            self._graph = Neo4jGraph(
                url=NEO4J_BOLT_URL,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
            )

            detected_version = "connected"
            try:
                version_rows = self._graph.query(
                    "CALL dbms.components() YIELD versions RETURN versions[0] as version"
                )
                if version_rows:
                    detected_version = version_rows[0].get("version", "unknown")
            except GqlError:
                pass

            return Neo4jConnectionResult(connected=True, version=detected_version)

        except (GqlError, ConnectionError, OSError):
            return Neo4jConnectionResult()

    def query_database_stats(self) -> DatabaseStats:
        """Gather node, relationship, and label counts from the database."""
        if not self._graph and not self.check_connection().connected:
            return DatabaseStats(error_message="Not connected to Neo4j")

        try:
            node_count_rows = self._graph.query("MATCH (n) RETURN count(n) as count")
            total_nodes = node_count_rows[0].get("count", 0) if node_count_rows else 0

            relationship_count_rows = self._graph.query("MATCH ()-[r]->() RETURN count(r) as count")
            total_relationships = relationship_count_rows[0].get("count", 0) if relationship_count_rows else 0

            label_count_rows = self._graph.query("""
                MATCH (n)
                UNWIND labels(n) as label
                RETURN label, count(*) as count
                ORDER BY count DESC
            """)
            node_types: dict[str, int] = {}
            if label_count_rows:
                for row in label_count_rows:
                    label_name = row.get("label")
                    label_count = row.get("count", 0)
                    if label_name:
                        node_types[label_name] = label_count

            programmer_count_rows = self._graph.query("MATCH (p:Person) RETURN count(p) as count")
            programmer_count = programmer_count_rows[0].get("count", 0) if programmer_count_rows else 0

            return DatabaseStats(
                total_nodes=total_nodes,
                total_relationships=total_relationships,
                node_types=node_types,
                programmer_count=programmer_count,
            )

        except GqlError as exc:
            return DatabaseStats(error_message=str(exc))

    def check_data_integrity(self) -> DataIntegrityReport:
        """Detect orphaned nodes and persons without skills."""
        if not self._graph and not self.check_connection().connected:
            return DataIntegrityReport()

        collected_warnings: list[str] = []

        try:
            orphan_rows = self._graph.query("""
                MATCH (n)
                WHERE NOT (n)-[]-()
                RETURN count(n) as orphaned_count
            """)
            orphaned_count = orphan_rows[0].get("orphaned_count", 0) if orphan_rows else 0
            if orphaned_count > 0:
                collected_warnings.append(f"{orphaned_count} orphaned nodes (no relationships)")

            skillless_rows = self._graph.query("""
                MATCH (p:Person)
                WHERE NOT (p)-[:HAS_SKILL]->()
                RETURN count(p) as count
            """)
            skillless_count = skillless_rows[0].get("count", 0) if skillless_rows else 0
            if skillless_count > 0:
                collected_warnings.append(f"{skillless_count} Person nodes without skills")

        except GqlError:
            pass

        return DataIntegrityReport(warnings=collected_warnings)

    def clear_all_data(self) -> bool:
        """Delete every node and relationship from the database."""
        if not self._graph and not self.check_connection().connected:
            print("❌ Cannot clear database: not connected")
            return False

        try:
            self._graph.query("MATCH (n) DETACH DELETE n")
            print("Database cleared successfully")
            return True
        except GqlError as exc:
            print(f"❌ Error clearing database: {exc}")
            return False


class GraphRAGSetup:
    """
    Main orchestrator for the GraphRAG educational setup.

    Handles environment initialization, Neo4j lifecycle management,
    data population, and interactive menu-driven workflows.
    """

    def __init__(self, *, mode: SetupMode = SetupMode.INTERACTIVE, learning_mode: bool = False) -> None:
        self.mode = mode
        self.learning_mode = learning_mode
        self.neo4j_checker = Neo4jStatusChecker()

    def run(self) -> bool:
        """Execute the full setup workflow according to the selected mode."""
        self._print_header()

        if not self._verify_prerequisites():
            return False

        current_state = self._analyze_current_state()
        if not current_state:
            return False

        match self.mode:
            case SetupMode.CHECK:
                self._display_status_report(current_state)
                return True
            case SetupMode.INIT:
                return self._execute_init(current_state)
            case SetupMode.FRESH:
                return self._execute_fresh_setup(current_state)
            case SetupMode.CONTINUE:
                return self._execute_continue(current_state)
            case _:
                return self._run_interactive_menu(current_state)

    def _print_header(self) -> None:
        print_section_header("🎓 GraphRAG Educational Setup System")

        learning_line = "\n📖 Learning Mode: Explanations will be provided for each step" if self.learning_mode else ""
        print(textwrap.dedent(f"""\
            {learning_line}
            🔧 Mode: {self.mode.value.title()}
            {'-' * 40}"""))

    def _verify_prerequisites(self) -> bool:
        """
        Verify that all required packages, API keys, and infrastructure
        (Docker) are available before proceeding with setup.
        """
        print("\n🔍 Checking Prerequisites...")

        all_prerequisites_met = True

        if self.learning_mode:
            print(textwrap.dedent("""\

                📚 What we're checking:
                  - Python packages (Azure OpenAI, Neo4j, etc.)
                  - Docker availability
                  - Neo4j database connection
                  - Azure OpenAI API key
            """))

        required_packages = ["openai", "neo4j", "langchain_openai", "langchain_neo4j"]
        missing_packages = [
            pkg for pkg in required_packages
            if importlib.util.find_spec(pkg) is None
        ]

        if missing_packages:
            print(textwrap.dedent(f"""\
                ✗ Missing Python packages: {', '.join(missing_packages)}
                  Run: uv sync"""))
            all_prerequisites_met = False
        else:
            print("✓ Required Python packages installed")

        azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_openai_api_key:
            print(textwrap.dedent("""\
                ✗ AZURE_OPENAI_API_KEY not found
                  Add your API key to .env file"""))
            all_prerequisites_met = False
        else:
            print("✓ Azure OpenAI API key found")

        docker_neo4j_status = detect_docker_neo4j_status()
        if not docker_neo4j_status.docker_available:
            print(textwrap.dedent("""\
                ✗ Docker not available
                  Install Docker Desktop and start it"""))
            all_prerequisites_met = False
        else:
            print("✓ Docker available")

        if not all_prerequisites_met:
            print("\n❌ Prerequisites not met. Please fix the issues above.")

        return all_prerequisites_met

    def _analyze_current_state(self) -> SystemState | None:
        """
        Connect to Neo4j (starting a container if necessary),
        then collect database statistics and integrity information.
        """
        print("\n🔍 Analyzing Current State...")

        if self.learning_mode:
            print(textwrap.dedent("""\

                📚 What's happening:
                  - Checking if Neo4j is running
                  - Looking for existing data
                  - Assessing data quality
            """))

        connection = self.neo4j_checker.check_connection()

        if not connection.connected:
            print("⚠️  Neo4j not running")
            current_docker_status = detect_docker_neo4j_status()

            if current_docker_status.neo4j_container_id:
                if not current_docker_status.container_running:
                    print("  Starting existing Neo4j container...")
                    self._start_existing_neo4j_container()
                    time.sleep(5)
                    connection = self.neo4j_checker.check_connection()
            else:
                print("  No Neo4j container found")
                if not self._provision_neo4j_via_docker():
                    return None
                connection = self.neo4j_checker.check_connection()

        if not connection.connected:
            print("❌ Could not establish Neo4j connection")
            self._print_docker_troubleshooting()
            return None

        print(f"✓ Connected to Neo4j {connection.version}")

        db_stats = self.neo4j_checker.query_database_stats()
        if db_stats.error_message:
            print(f"⚠️  Could not get database stats: {db_stats.error_message}")
            db_stats = DatabaseStats()

        integrity_report = self.neo4j_checker.check_data_integrity()

        return SystemState(
            connection=connection,
            stats=db_stats,
            integrity=integrity_report,
            has_data=db_stats.total_nodes > 0,
            has_programmers=db_stats.programmer_count > 0,
        )

    @staticmethod
    def _display_status_report(state: SystemState) -> None:
        """Print a comprehensive overview of the current knowledge graph."""
        print(textwrap.dedent(f"""\

            📊 Database Status Report
            {'=' * 40}"""))

        if state.has_data:
            entity_lines = "\n".join(
                f"  • {entity_label}: {entity_count:,}"
                for entity_label, entity_count in state.stats.node_types.items()
                if entity_label and entity_count > 0
            )
            print(textwrap.dedent(f"""\
                ✓ Knowledge graph exists
                  📈 Total nodes: {state.stats.total_nodes:,}
                  🔗 Total relationships: {state.stats.total_relationships:,}

                📋 Entity Counts:
                {entity_lines}"""))

            if state.integrity.issues:
                issue_lines = "\n".join(f"  • {desc}" for desc in state.integrity.issues)
                print(textwrap.dedent(f"""\

                    ⚠️  Data Issues:
                    {issue_lines}"""))

            if state.integrity.warnings:
                warning_lines = "\n".join(f"  • {desc}" for desc in state.integrity.warnings)
                print(textwrap.dedent(f"""\

                    💡 Suggestions:
                    {warning_lines}"""))
        else:
            print("✗ No data found (empty database)")

        print(textwrap.dedent(f"""\

            🔗 Connection Info:
              URI: {state.connection.uri}
              Version: {state.connection.version}"""))

    def _run_interactive_menu(self, state: SystemState) -> bool:
        """Present an interactive menu and execute the chosen action."""
        print(textwrap.dedent(f"""\

            🎯 What would you like to do?
            {'-' * 40}"""))

        if state.has_data:
            menu_options = [
                ("1", "Continue with existing data", "continue"),
                ("2", "View current data status", "status"),
                ("3", "Add more data to existing graph", "extend"),
                ("4", "Rebuild from scratch", "fresh"),
                ("5", "Run test queries", "test"),
                ("6", "Backup current data", "backup"),
                ("0", "Exit", "exit"),
            ]
        else:
            menu_options = [
                ("1", "Initialize new knowledge graph", "fresh"),
                ("2", "Load sample data", "sample"),
                ("3", "View setup instructions", "help"),
                ("0", "Exit", "exit"),
            ]

        for option_key, option_description, _ in menu_options:
            print(f"  {option_key}. {option_description}")

        while True:
            user_choice = input("\nEnter your choice (0-6): ").strip()

            matched_action = next(
                (action for key, _, action in menu_options if key == user_choice),
                None,
            )
            if matched_action:
                return self._dispatch_menu_action(action_name=matched_action, state=state)
            print("Invalid choice. Please try again.")

    def _dispatch_menu_action(self, *, action_name: str, state: SystemState) -> bool:
        """Route a menu selection to the appropriate handler."""
        match action_name:
            case "exit":
                print("👋 Goodbye!")
                return True
            case "continue":
                print("\n✓ Using existing knowledge graph")
                self._print_next_steps(state)
                return True
            case "status":
                self._display_status_report(state)
                return True
            case "fresh":
                return self._execute_fresh_setup(state)
            case "extend":
                return self._extend_existing_data()
            case "test":
                return self._run_test_queries()
            case "backup":
                return self._backup_data()
            case "sample":
                return self._load_sample_data()
            case "help":
                self._print_help()
                return True
            case _:
                print(f"Action '{action_name}' not implemented yet")
                return True

    def _execute_fresh_setup(self, state: SystemState) -> bool:
        """
        Clear the database, generate sample data, build the knowledge graph,
        and verify the result. The full pipeline for a clean start.
        """
        if state.has_data:
            print("\n⚠️  Warning: This will delete all existing data!")
            if not self._ask_for_confirmation("Continue with fresh setup?"):
                return True

        print("\n🚀 Setting up fresh GraphRAG system...")

        if self.learning_mode:
            print(textwrap.dedent("""\

                📚 What's happening:
                  1. Clearing database completely
                  2. Generating sample programmer data
                  3. Creating knowledge graph from data
                  4. Verifying the setup
            """))

        print("\n1️⃣ Clearing database...")
        if not self.neo4j_checker.clear_all_data():
            print("❌ Failed to clear database")
            return False
        print("✓ Database cleared")

        print("\n2️⃣ Creating project structure...")
        self._create_project_directories()

        print("\n3️⃣ Generating sample data...")
        if not self._run_child_script("1_generate_data.py"):
            return False

        print("\n4️⃣ Building knowledge graph...")
        if not self._run_child_script("2_data_to_knowledge_graph.py"):
            return False

        print("\n5️⃣ Verifying setup...")
        verified_state = self._analyze_current_state()
        if verified_state and verified_state.has_data:
            print("✅ Setup completed successfully!")
            self._display_status_report(verified_state)
            self._print_next_steps(verified_state)
            return True

        print("❌ Setup verification failed")
        return False

    def _execute_continue(self, state: SystemState) -> bool:
        """Use the existing knowledge graph, or fall back to a fresh setup if empty."""
        if not state.has_data:
            print("No existing data found. Initializing fresh setup...")
            return self._execute_fresh_setup(state)

        print("✓ Using existing knowledge graph")
        self._print_next_steps(state)
        return True

    def _execute_init(self, state: SystemState) -> bool:
        """
        Initialize only the Neo4j server and project directories.
        The database remains empty for manual data population.
        """
        print("\n🔧 Initializing environment...")

        if self.learning_mode:
            print(textwrap.dedent("""\

                📚 What's happening:
                  1. Starting Neo4j server if not running
                  2. Creating project directory structure
                  3. Database will remain empty for manual data population
            """))

        if not state.connection.connected:
            print("\n🚀 Starting Neo4j server...")
            init_docker_status = detect_docker_neo4j_status()

            if init_docker_status.neo4j_container_id and not init_docker_status.container_running:
                self._start_existing_neo4j_container()
                print("⏳ Waiting for Neo4j to be ready...")
                time.sleep(5)
            elif not init_docker_status.neo4j_container_id:
                if not self._provision_neo4j_via_docker():
                    print("❌ Failed to start Neo4j")
                    return False

            verified_connection = self.neo4j_checker.check_connection()
            if not verified_connection.connected:
                print("❌ Could not establish Neo4j connection")
                self._print_docker_troubleshooting()
                return False

        print("\n📁 Creating project structure...")
        self._create_project_directories()

        print(textwrap.dedent(f"""\

            🌐 Neo4j Server Active:
              📊 Neo4j Browser: {NEO4J_BROWSER_URL}
              🔌 Bolt Protocol:  {NEO4J_BOLT_URL}
              👤 Username: {NEO4J_USERNAME}
              🔑 Password: {NEO4J_PASSWORD}"""))

        if state.has_data:
            print(textwrap.dedent(f"""\

                📊 Current Database Status:
                  ⚠️  Database contains {state.stats.total_nodes} nodes
                  💡 Use --fresh to clear and rebuild, or --continue to use existing data"""))
        else:
            print(textwrap.dedent("""\

                📊 Current Database Status:
                  ✓ Database is empty and ready"""))

        print(textwrap.dedent("""\

            🎯 Next Steps:
              1. Generate sample CVs:
                 uv run 1_generate_data.py
              2. Build knowledge graph:
                 uv run 2_data_to_knowledge_graph.py
              3. Query the graph:
                 uv run 3_query_knowledge_graph.py

            ✅ Initialization complete!"""))

        return True

    def _run_child_script(self, script_name: str) -> bool:
        """Execute a companion Python script as a subprocess."""
        try:
            if self.learning_mode:
                print(f"  📝 Running {script_name}...")

            subprocess_result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if subprocess_result.returncode == 0:
                print(f"✓ {script_name} completed successfully")
                return True

            print(textwrap.dedent(f"""\
                ❌ {script_name} failed:
                Error: {subprocess_result.stderr}"""))
            return False

        except subprocess.TimeoutExpired:
            print(f"❌ {script_name} timed out")
            return False
        except OSError as exc:
            print(f"❌ Error running {script_name}: {exc}")
            return False

    @staticmethod
    def _create_project_directories() -> None:
        """Ensure all required project directories exist."""
        for directory_path in PROJECT_DIRECTORIES:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
        print("✓ Project structure created")

    @staticmethod
    def _provision_neo4j_via_docker() -> bool:
        """Create a docker-compose file if missing and start Neo4j."""
        print("\n🐳 Setting up Neo4j with Docker...")

        compose_file = Path("docker-compose.yml")
        if not compose_file.exists():
            print("Creating docker-compose.yml...")
            compose_file.write_text(DOCKER_COMPOSE_TEMPLATE)
            print("✓ Created docker-compose.yml")

        try:
            compose_result = subprocess.run(
                ["docker-compose", "up", "-d"],
                capture_output=True,
                text=True,
            )

            if compose_result.returncode == 0:
                print(textwrap.dedent("""\
                    ✓ Neo4j started with Docker Compose
                    ⏳ Waiting for Neo4j to be ready..."""))
                time.sleep(15)
                return True

            print(f"❌ Failed to start Neo4j: {compose_result.stderr}")
            return False

        except OSError as exc:
            print(f"❌ Error starting Neo4j: {exc}")
            return False

    @staticmethod
    def _start_existing_neo4j_container() -> None:
        """Restart a previously created but stopped Neo4j container."""
        try:
            existing_container_status = detect_docker_neo4j_status()
            if existing_container_status.neo4j_container_id:
                start_result = subprocess.run(
                    ["docker", "start", existing_container_status.neo4j_container_id],
                    capture_output=True,
                    text=True,
                )
                if start_result.returncode == 0:
                    print("✓ Neo4j container started")
                else:
                    print(f"⚠️  Could not start container: {start_result.stderr}")
        except OSError as exc:
            print(f"⚠️  Error starting container: {exc}")

    @staticmethod
    def _print_next_steps(state: SystemState) -> None:
        """Print context-aware guidance for what to do next."""
        if state.has_programmers:
            query_guidance = textwrap.dedent("""\
                • Try some queries:
                  python 3_query_knowledge_graph.py
                • Test GraphRAG vs Naive RAG:
                  python 5_compare_systems.py""")
        else:
            query_guidance = textwrap.dedent("""\
                • Generate some data first:
                  python 1_generate_data.py""")

        print(textwrap.dedent(f"""\

            🎯 Next Steps:
            {'-' * 30}
            {query_guidance}
            • Explore in Neo4j Browser:
              {NEO4J_BROWSER_URL}
              Username: {NEO4J_USERNAME}, Password: {NEO4J_PASSWORD}"""))

    @staticmethod
    def _print_help() -> None:
        """Display a detailed overview of the GraphRAG project and its files."""
        print(textwrap.dedent(f"""\

            📚 GraphRAG Setup Help
            {'=' * 30}

            This system demonstrates GraphRAG (Graph Retrieval Augmented Generation)
            for a programmer staffing use case.

            Components:
            • Neo4j: Graph database for storing relationships
            • Azure OpenAI: LLM for text processing and queries
            • LangChain: Framework connecting LLMs to data

            Workflow:
            1. Generate synthetic programmer CVs and project data
            2. Extract knowledge graph from unstructured PDFs
            3. Query the graph using natural language
            4. Compare with traditional RAG approaches

            Files:
            • 1_generate_data.py: Create sample CVs and projects
            • 2_data_to_knowledge_graph.py: Build graph from PDFs
            • 3_query_knowledge_graph.py: Query the graph
            • 4_graph_rag_system.py: Full GraphRAG implementation
            • 5_compare_systems.py: Compare different approaches
        """))

    @staticmethod
    def _print_docker_troubleshooting() -> None:
        """Print common Docker troubleshooting steps."""
        print(textwrap.dedent("""\

            🔧 Docker Troubleshooting:
            1. Make sure Docker Desktop is running
            2. Try: docker-compose up -d
            3. Check logs: docker-compose logs neo4j
            4. Reset: docker-compose down && docker-compose up -d
        """))

    @staticmethod
    def _ask_for_confirmation(prompt_message: str) -> bool:
        """Ask the user for yes/no confirmation."""
        while True:
            user_response = input(f"{prompt_message} (y/N): ").strip().lower()
            if user_response in ("y", "yes"):
                return True
            if user_response in ("n", "no", ""):
                return False
            print("Please enter 'y' or 'n'")

    @staticmethod
    def _extend_existing_data() -> bool:
        """Placeholder for extending an existing knowledge graph with new data."""
        print(textwrap.dedent("""\

            📈 Extending existing knowledge graph...
            This feature is coming soon!"""))
        return True

    def _run_test_queries(self) -> bool:
        """Execute the query script to validate the knowledge graph contents."""
        print("\n🧪 Running test queries...")
        return self._run_child_script("3_query_knowledge_graph.py")

    @staticmethod
    def _backup_data() -> bool:
        """Placeholder for exporting the current graph state to a backup file."""
        print(textwrap.dedent("""\

            💾 Backing up data...
            This feature is coming soon!"""))
        return True

    def _load_sample_data(self) -> bool:
        """Generate sample programmer CVs and project descriptions."""
        print("\n📦 Loading sample data...")
        return self._run_child_script("1_generate_data.py")


def parse_command_line_arguments() -> ParsedSetupConfig:
    """Parse CLI flags and return a structured setup configuration."""
    argument_parser = argparse.ArgumentParser(
        description="GraphRAG Educational Setup System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 0_setup.py                # Interactive mode
  python 0_setup.py --init          # Initialize Neo4j and directories only
  python 0_setup.py --fresh         # Fresh installation
  python 0_setup.py --continue      # Use existing data
  python 0_setup.py --check         # Check status only
  python 0_setup.py --learning      # Educational mode with explanations
        """,
    )

    argument_parser.add_argument("--init", action="store_true",
                                 help="Initialize Neo4j and project structure only")
    argument_parser.add_argument("--fresh", action="store_true",
                                 help="Force fresh setup (clears existing data)")
    argument_parser.add_argument("--continue", action="store_true", dest="continue_mode",
                                 help="Continue with existing data")
    argument_parser.add_argument("--check", action="store_true",
                                 help="Check status only")
    argument_parser.add_argument("--learning", action="store_true",
                                 help="Enable educational explanations")

    parsed_args = argument_parser.parse_args()

    if parsed_args.init:
        selected_mode = SetupMode.INIT
    elif parsed_args.fresh:
        selected_mode = SetupMode.FRESH
    elif parsed_args.continue_mode:
        selected_mode = SetupMode.CONTINUE
    elif parsed_args.check:
        selected_mode = SetupMode.CHECK
    elif parsed_args.learning:
        selected_mode = SetupMode.LEARNING
    else:
        selected_mode = SetupMode.INTERACTIVE

    return ParsedSetupConfig(mode=selected_mode, learning_mode=parsed_args.learning)


def main() -> None:
    try:
        parsed_config = parse_command_line_arguments()
        setup = GraphRAGSetup(mode=parsed_config.mode, learning_mode=parsed_config.learning_mode)
        setup_succeeded = setup.run()

        if not setup_succeeded:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
        sys.exit(0)
    except (OSError, subprocess.SubprocessError, GqlError) as known_error:
        print(f"\n❌ Setup error: {known_error}")
        sys.exit(1)
    except Exception as unexpected_error:
        print(f"\n❌ Unexpected error: {unexpected_error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
