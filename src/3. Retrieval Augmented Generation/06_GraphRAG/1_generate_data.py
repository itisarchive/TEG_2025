"""
🏗️ GraphRAG Data Generation — Programmer Profiles, CV PDFs, Projects, and RFPs
================================================================================

Generates realistic programmer profiles and PDF CVs for GraphRAG educational demonstration.
Uses Azure OpenAI LLM to create unique, unstructured CVs in markdown format, then converts
them to PDF. Also produces project records with skill-based programmer assignments and
RFP (Request for Proposal) documents.

The generated artifacts feed the downstream knowledge-graph pipeline
(2_data_to_knowledge_graph.py) and the naive-RAG baseline (4_naive_rag_cv.py).

Prerequisites:
    - Azure OpenAI credentials in environment / .env
    - Python 3.13+ with dependencies from pyproject.toml
    - Configuration in utils/config.toml
"""

import json
import random
import textwrap
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import markdown
import toml
from dotenv import load_dotenv
from faker import Faker
from langchain_openai import AzureChatOpenAI
from weasyprint import HTML, CSS

FAKE_DATA_FACTORY = Faker()

PROFICIENCY_RANK: dict[str, int] = {
    "Beginner": 1,
    "Intermediate": 2,
    "Advanced": 3,
    "Expert": 4,
}

ALL_PROGRAMMING_SKILLS: list[str] = [
    "Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust",
    "React", "Vue.js", "Angular", "Node.js", "Django", "Flask", "FastAPI",
    "PostgreSQL", "MongoDB", "Redis", "MySQL",
    "AWS", "Docker", "Kubernetes", "Jenkins", "Git",
    "Machine Learning", "Data Science", "DevOps", "Microservices",
]

PROJECT_TYPE_NAMES: list[str] = [
    "E-commerce Platform", "Data Analytics Dashboard", "Mobile App Development",
    "API Gateway Implementation", "Machine Learning Pipeline", "Web Application",
    "Microservices Architecture", "Real-time Chat System", "Content Management System",
    "Payment Processing System", "DevOps Automation", "Cloud Migration",
    "Security Audit System", "Inventory Management", "Customer Portal",
]

CLIENT_COMPANY_NAMES: list[str] = [
    "TechCorp", "DataSystems Inc", "CloudNative Solutions", "FinTech Innovations",
    "HealthTech Partners", "RetailMax", "LogisticsPro", "EduTech Solutions",
    "MediaStream", "GreenEnergy Co", "SmartCity Initiative", "BioTech Labs",
]

CERTIFICATION_NAMES: list[str] = [
    "AWS Certified Solutions Architect",
    "Google Cloud Professional",
    "Certified Kubernetes Administrator",
    "Microsoft Azure Developer",
    "Scrum Master Certification",
    "Docker Certified Associate",
]

PROJECT_STATUS_OPTIONS: list[str] = ["completed", "active", "planned", "on_hold"]
PROJECT_STATUS_WEIGHTS: list[int] = [50, 30, 15, 5]

PROGRAMMER_PROJECT_EXPERIENCE_POOL: list[str] = [
    "E-commerce Platform", "Data Analytics Dashboard", "Mobile App",
    "API Gateway", "Machine Learning Pipeline", "Web Application",
    "Microservices Architecture", "Real-time Chat System",
    "Content Management System", "Payment Processing System",
]

RFP_PROJECT_TYPE_NAMES: list[str] = [
    "Enterprise Web Application", "Mobile App Development", "Data Analytics Platform",
    "Cloud Migration Project", "E-commerce Modernization", "API Integration Platform",
    "Machine Learning Implementation", "DevOps Automation", "Security Enhancement",
]

RFP_CLIENT_NAMES: list[str] = [
    "Global Finance Corp", "MedTech Industries", "Retail Solutions Ltd",
    "Manufacturing Plus", "Education Network", "Energy Systems Co",
]

RFP_BUDGET_RANGE_LABELS: list[str] = [
    "$100K - $250K", "$250K - $500K", "$500K - $1M", "$1M - $2M",
]

RFP_SKILL_POOL: list[str] = [
    "Python", "JavaScript", "TypeScript", "Java", "React", "Angular",
    "Node.js", "Django", "AWS", "Docker", "Kubernetes", "PostgreSQL",
    "MongoDB", "Machine Learning", "DevOps", "Microservices",
]

RFP_PREFERRED_CERTIFICATIONS: list[str] = [
    "AWS Certified Solutions Architect",
    "Google Cloud Professional",
    "Certified Kubernetes Administrator",
]

CV_PDF_STYLESHEET = """\
body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    max-width: 800px;
    margin: 40px auto;
    padding: 20px;
}
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
h2 { color: #34495e; margin-top: 30px; }
h3 { color: #7f8c8d; }
strong { color: #2c3e50; }
ul { margin-left: 20px; }
"""


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


@dataclass(frozen=True)
class ProgrammerSkill:
    """Single skill with its proficiency level (e.g. 'Python', 'Advanced')."""
    skill_name: str
    proficiency: str


@dataclass(frozen=True)
class ProgrammerProfile:
    """Structured profile for a generated programmer."""
    profile_id: int
    full_name: str
    email: str
    city: str
    skills: tuple[ProgrammerSkill, ...]
    project_names: tuple[str, ...]
    certifications: tuple[str, ...]

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "id": self.profile_id,
            "name": self.full_name,
            "email": self.email,
            "location": self.city,
            "skills": [
                {"name": skill.skill_name, "proficiency": skill.proficiency}
                for skill in self.skills
            ],
            "projects": list(self.project_names),
            "certifications": list(self.certifications),
        }


@dataclass(frozen=True)
class SkillRequirement:
    """A single skill requirement for a project or RFP."""
    skill_name: str
    min_proficiency: str
    is_mandatory: bool


@dataclass(frozen=True)
class ProgrammerAssignment:
    """Records that a programmer is assigned to a project in a date range."""
    programmer_name: str
    programmer_id: int
    assignment_start_date: str
    assignment_end_date: str


@dataclass
class ProjectRecord:
    """A generated project with requirements and team assignments."""
    project_id: str
    project_name: str
    client: str
    description: str
    start_date: str
    end_date: str | None
    estimated_duration_months: int
    budget: int | None
    status: str
    team_size: int
    requirements: list[SkillRequirement]
    assigned_programmers: list[ProgrammerAssignment] = field(default_factory=list)

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "id": self.project_id,
            "name": self.project_name,
            "client": self.client,
            "description": self.description,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "estimated_duration_months": self.estimated_duration_months,
            "budget": self.budget,
            "status": self.status,
            "team_size": self.team_size,
            "requirements": [
                {
                    "skill_name": req.skill_name,
                    "min_proficiency": req.min_proficiency,
                    "is_mandatory": req.is_mandatory,
                }
                for req in self.requirements
            ],
            "assigned_programmers": [
                {
                    "programmer_name": assignment.programmer_name,
                    "programmer_id": assignment.programmer_id,
                    "assignment_start_date": assignment.assignment_start_date,
                    "assignment_end_date": assignment.assignment_end_date,
                }
                for assignment in self.assigned_programmers
            ],
        }


@dataclass(frozen=True)
class RfpRequirement:
    """A single skill requirement inside an RFP document."""
    skill_name: str
    min_proficiency: str
    is_mandatory: bool
    preferred_certifications: tuple[str, ...]


@dataclass(frozen=True)
class RfpRecord:
    """A generated Request for Proposal."""
    rfp_id: str
    title: str
    client: str
    description: str
    project_type: str
    duration_months: int
    team_size: int
    budget_range: str
    start_date: str
    requirements: tuple[RfpRequirement, ...]
    location: str
    remote_allowed: bool

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "id": self.rfp_id,
            "title": self.title,
            "client": self.client,
            "description": self.description,
            "project_type": self.project_type,
            "duration_months": self.duration_months,
            "team_size": self.team_size,
            "budget_range": self.budget_range,
            "start_date": self.start_date,
            "requirements": [
                {
                    "skill_name": req.skill_name,
                    "min_proficiency": req.min_proficiency,
                    "is_mandatory": req.is_mandatory,
                    "preferred_certifications": list(req.preferred_certifications),
                }
                for req in self.requirements
            ],
            "location": self.location,
            "remote_allowed": self.remote_allowed,
        }


def load_generation_config(config_path: str = "utils/config.toml") -> dict[str, Any]:
    """Load the TOML configuration that drives every generation parameter."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_file.open("r", encoding="utf-8") as file_handle:
        return toml.load(file_handle)


def build_random_skills(proficiency_weights: list[int]) -> tuple[ProgrammerSkill, ...]:
    """Pick 5–12 random skills and assign weighted proficiency levels."""
    proficiency_level_names = list(PROFICIENCY_RANK.keys())
    skills_count = random.randint(5, 12)
    chosen_skill_names = random.sample(ALL_PROGRAMMING_SKILLS, skills_count)

    return tuple(
        ProgrammerSkill(
            skill_name=chosen_name,
            proficiency=random.choices(proficiency_level_names, weights=proficiency_weights)[0],
        )
        for chosen_name in chosen_skill_names
    )


def pick_random_project_names() -> tuple[str, ...]:
    """Pick 2–5 random project-type names for a programmer's experience."""
    return tuple(random.sample(PROGRAMMER_PROJECT_EXPERIENCE_POOL, random.randint(2, 5)))


def pick_random_certifications() -> tuple[str, ...]:
    """Pick 0–3 random certifications."""
    certifications_count = random.randint(0, 3)
    if certifications_count == 0:
        return ()
    return tuple(random.sample(CERTIFICATION_NAMES, certifications_count))


class GraphRAGDataGenerator:
    """
    Integrated generator for programmer profiles, PDF CVs, project records,
    and RFP documents.

    All generated artifacts are designed to feed a downstream GraphRAG pipeline:
    profiles and CVs represent the "people" dimension, projects encode
    skill-based assignments, and RFPs model future staffing demands.
    """

    def __init__(self, config_path: str = "utils/config.toml") -> None:
        self.config = load_generation_config(config_path)
        self.llm = AzureChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    def generate_programmer_profiles(
            self,
            profiles_count: int,
    ) -> list[ProgrammerProfile]:
        """
        Generate *profiles_count* realistic programmer profiles with random skills.

        Each profile receives a unique combination of 5–12 programming skills
        (with weighted proficiency distribution), 2–5 past project names, and
        0–3 industry certifications — all drawn from the module-level pools.
        """
        if profiles_count <= 0:
            raise ValueError("Number of profiles must be positive")

        proficiency_weights: list[int] = self.config["skills"]["proficiency_weights"]

        return [
            ProgrammerProfile(
                profile_id=index + 1,
                full_name=FAKE_DATA_FACTORY.name(),
                email=FAKE_DATA_FACTORY.email(),
                city=FAKE_DATA_FACTORY.city(),
                skills=build_random_skills(proficiency_weights),
                project_names=pick_random_project_names(),
                certifications=pick_random_certifications(),
            )
            for index in range(profiles_count)
        ]

    def generate_project_records(
            self,
            projects_count: int = 20,
            programmer_profiles: list[ProgrammerProfile] | None = None,
    ) -> list[ProjectRecord]:
        """
        Generate realistic project records, optionally assigning programmers by skill match.

        When *programmer_profiles* are provided, the generator uses config-driven
        requirement ranges and attempts to assign qualifying programmers to each
        active or completed project based on mandatory skill matching and
        availability windows.
        """
        if projects_count <= 0:
            raise ValueError("Number of projects must be positive")

        available_skill_names = self._collect_skill_names_from_profiles(programmer_profiles)

        project_records: list[ProjectRecord] = [
            self._build_single_project(
                project_index=project_index,
                available_skill_names=available_skill_names,
                has_programmer_profiles=(programmer_profiles is not None),
            )
            for project_index in range(projects_count)
        ]

        if programmer_profiles is not None:
            project_records = self._assign_programmers_to_projects(
                project_records, programmer_profiles,
            )

        return project_records

    @staticmethod
    def _collect_skill_names_from_profiles(
            profiles: list[ProgrammerProfile] | None,
    ) -> list[str]:
        if profiles is None:
            return list(ALL_PROGRAMMING_SKILLS)
        unique_skills: set[str] = {
            skill.skill_name
            for profile in profiles
            for skill in profile.skills
        }
        return list(unique_skills)

    def _build_single_project(
            self,
            *,
            project_index: int,
            available_skill_names: list[str],
            has_programmer_profiles: bool,
    ) -> ProjectRecord:
        project_start_date = FAKE_DATA_FACTORY.date_between(start_date="-2y", end_date="+6m")
        duration_months = random.randint(3, 18)

        chosen_status = random.choices(PROJECT_STATUS_OPTIONS, weights=PROJECT_STATUS_WEIGHTS)[0]
        computed_end_date: str | None = None
        if chosen_status == "completed":
            computed_end_date = (project_start_date + timedelta(days=duration_months * 30)).isoformat()

        skill_requirements = self._build_skill_requirements(
            available_skill_names=available_skill_names,
            use_config_ranges=has_programmer_profiles,
        )

        return ProjectRecord(
            project_id=f"PRJ-{project_index + 1:03d}",
            project_name=f"{random.choice(PROJECT_TYPE_NAMES)} for {random.choice(CLIENT_COMPANY_NAMES)}",
            client=random.choice(CLIENT_COMPANY_NAMES),
            description=(
                f"Development of {random.choice(PROJECT_TYPE_NAMES).lower()} "
                f"with focus on scalability and performance"
            ),
            start_date=project_start_date.isoformat(),
            end_date=computed_end_date,
            estimated_duration_months=duration_months,
            budget=random.randint(50_000, 500_000) if random.choice([True, False]) else None,
            status=chosen_status,
            team_size=random.randint(2, 8),
            requirements=skill_requirements,
        )

    def _build_skill_requirements(
            self,
            *,
            available_skill_names: list[str],
            use_config_ranges: bool,
    ) -> list[SkillRequirement]:
        if use_config_ranges:
            requirements_config = self.config["project_requirements"]
            requirements_count = random.randint(
                requirements_config["min_requirements"],
                requirements_config["max_requirements"],
            )
            mandatory_probability: float = requirements_config["mandatory_probability"]
            proficiency_choices: list[str] = self.config["skills"]["proficiency_levels"]
        else:
            requirements_count = random.randint(3, 8)
            mandatory_probability = 2 / 3
            proficiency_choices = list(PROFICIENCY_RANK.keys())

        chosen_skills = random.sample(
            available_skill_names,
            min(requirements_count, len(available_skill_names)),
        )

        return [
            SkillRequirement(
                skill_name=chosen_skill,
                min_proficiency=random.choice(proficiency_choices),
                is_mandatory=(random.random() < mandatory_probability),
            )
            for chosen_skill in chosen_skills
        ]

    def _assign_programmers_to_projects(
            self,
            project_records: list[ProjectRecord],
            programmer_profiles: list[ProgrammerProfile],
    ) -> list[ProjectRecord]:
        """Assign programmers to active/completed projects based on skill matching."""
        assignments_by_programmer_id: dict[int, list[ProgrammerAssignment]] = {
            profile.profile_id: [] for profile in programmer_profiles
        }

        assignable_projects = [
            project for project in project_records
            if project.status in ("active", "completed")
        ]
        assignment_probability: float = self.config["assignment"]["assignment_probability"]

        for project in assignable_projects:
            if random.random() > assignment_probability:
                continue

            mandatory_requirements = [req for req in project.requirements if req.is_mandatory]
            max_team_members = min(project.team_size, len(programmer_profiles))

            eligible_programmers = [
                candidate for candidate in programmer_profiles
                if (
                        self._meets_mandatory_requirements(candidate, mandatory_requirements)
                        and self._is_available_for_project(
                    project_start=project.start_date,
                    project_end=project.end_date,
                    existing_assignments=assignments_by_programmer_id[candidate.profile_id],
                )
                )
            ]

            selected_team = random.sample(
                eligible_programmers,
                min(max_team_members, len(eligible_programmers)),
            )

            for member in selected_team:
                computed_end_date = self._compute_assignment_end_date(project)
                new_assignment = ProgrammerAssignment(
                    programmer_name=member.full_name,
                    programmer_id=member.profile_id,
                    assignment_start_date=project.start_date,
                    assignment_end_date=computed_end_date,
                )
                project.assigned_programmers.append(new_assignment)
                assignments_by_programmer_id[member.profile_id].append(new_assignment)

        return project_records

    @staticmethod
    def _meets_mandatory_requirements(
            programmer: ProgrammerProfile,
            mandatory_requirements: list[SkillRequirement],
    ) -> bool:
        for requirement in mandatory_requirements:
            required_level = PROFICIENCY_RANK.get(requirement.min_proficiency, 0)
            requirement_satisfied = False
            for skill in programmer.skills:
                if skill.skill_name == requirement.skill_name:
                    programmer_level = PROFICIENCY_RANK.get(skill.proficiency, 0)
                    if programmer_level >= required_level:
                        requirement_satisfied = True
                    break
            if not requirement_satisfied:
                return False
        return True

    @staticmethod
    def _is_available_for_project(
            project_start: str,
            project_end: str | None,
            existing_assignments: list[ProgrammerAssignment],
    ) -> bool:
        new_project_start = datetime.fromisoformat(project_start).date()
        new_project_end = datetime.fromisoformat(project_end).date() if project_end else None

        for existing in existing_assignments:
            existing_start = datetime.fromisoformat(existing.assignment_start_date).date()
            existing_end = (
                datetime.fromisoformat(existing.assignment_end_date).date()
                if existing.assignment_end_date else None
            )

            if existing_end is None:
                if new_project_end is None or new_project_start <= existing_start:
                    return False
            elif new_project_end is None:
                if existing_end >= new_project_start:
                    return False
            else:
                if not (new_project_end < existing_start or new_project_start > existing_end):
                    return False
        return True

    def _compute_assignment_end_date(self, project: ProjectRecord) -> str:
        """Derive a realistic assignment end date from project dates and config."""
        days_before_min: int = self.config["assignment"]["assignment_end_days_before_min"]
        days_before_max: int = self.config["assignment"]["assignment_end_days_before_max"]

        project_start = datetime.fromisoformat(project.start_date).date()

        if project.end_date is not None:
            return self._end_date_clamped_to_project_duration(
                project_start=project_start,
                project_end=datetime.fromisoformat(project.end_date).date(),
                days_before_min=days_before_min,
                days_before_max=days_before_max,
            )

        estimated_end = project_start + timedelta(days=project.estimated_duration_months * 30)
        random_days_before = random.randint(days_before_min, days_before_max)
        return (estimated_end - timedelta(days=random_days_before)).isoformat()

    @staticmethod
    def _end_date_clamped_to_project_duration(
            *,
            project_start: date,
            project_end: date,
            days_before_min: int,
            days_before_max: int,
    ) -> str:
        total_duration_days = (project_end - project_start).days
        days_before = min(
            random.randint(days_before_min, days_before_max),
            max(1, total_duration_days - 1),
        )
        return (project_end - timedelta(days=days_before)).isoformat()

    @staticmethod
    def generate_rfp_records(rfps_count: int = 3) -> list[RfpRecord]:
        """
        Generate realistic RFP (Request for Proposal) records.

        Each RFP includes 4–10 technical skill requirements drawn from a pool of
        popular technologies, along with mandatory/optional flags and preferred
        certifications — simulating real-world procurement documents used in
        staffing decisions.
        """
        if rfps_count <= 0:
            raise ValueError("Number of RFPs must be positive")

        rfp_records: list[RfpRecord] = []
        for rfp_index in range(rfps_count):
            rfp_start_date = FAKE_DATA_FACTORY.date_between(start_date="+1m", end_date="+6m")
            requirements_count = random.randint(4, 10)
            chosen_rfp_skills = random.sample(RFP_SKILL_POOL, requirements_count)

            rfp_requirements = tuple(
                RfpRequirement(
                    skill_name=skill_name,
                    min_proficiency=random.choice(["Intermediate", "Advanced", "Expert"]),
                    is_mandatory=random.choice([True, True, False]),
                    preferred_certifications=tuple(random.sample(
                        RFP_PREFERRED_CERTIFICATIONS,
                        random.randint(0, 2),
                    )),
                )
                for skill_name in chosen_rfp_skills
            )

            chosen_rfp_type = random.choice(RFP_PROJECT_TYPE_NAMES)
            rfp_records.append(
                RfpRecord(
                    rfp_id=f"RFP-{rfp_index + 1:03d}",
                    title=f"{chosen_rfp_type} Development",
                    client=random.choice(RFP_CLIENT_NAMES),
                    description=f"Seeking experienced development team for {chosen_rfp_type.lower()}",
                    project_type=chosen_rfp_type,
                    duration_months=random.randint(6, 24),
                    team_size=random.randint(3, 12),
                    budget_range=random.choice(RFP_BUDGET_RANGE_LABELS),
                    start_date=rfp_start_date.isoformat(),
                    requirements=rfp_requirements,
                    location=FAKE_DATA_FACTORY.city(),
                    remote_allowed=random.choice([True, True, False]),
                )
            )

        return rfp_records

    def generate_cv_markdown_via_llm(self, profile: ProgrammerProfile) -> str:
        """
        Ask Azure OpenAI to produce a realistic, unique CV in markdown format.

        The LLM receives the programmer's structured profile (name, skills with
        proficiency levels, past projects, certifications) and returns a
        free-form CV document — intentionally unstructured to simulate real CVs
        that the downstream GraphRAG pipeline must parse and understand.
        """
        skills_description = ", ".join(
            f"{skill.skill_name} ({skill.proficiency})" for skill in profile.skills
        )

        cv_generation_prompt = textwrap.dedent(f"""\
            Create a professional CV in markdown format for a programmer with the following details:

            Name: {profile.full_name}
            Email: {profile.email}
            Location: {profile.city}
            Skills: {skills_description}
            Projects: {', '.join(profile.project_names)}
            Certifications: {', '.join(profile.certifications)}

            Requirements:
            1. Use proper markdown formatting (headers, lists, emphasis)
            2. Create realistic content with specific details and achievements
            3. Include sections like: Summary, Experience, Skills, Projects, Education, etc.
            4. Make it unique and personal - vary the structure and tone
            5. Add realistic company names, dates, and project descriptions
            6. Include specific metrics and achievements where appropriate
            7. IMPORTANT: Use the proficiency levels provided for each skill \
(Beginner, Intermediate, Advanced, Expert) in your skills sections

            Make each CV feel authentic and written by a real person, not a template.
            Use markdown syntax like # for headers, - for bullet points, **bold**, etc.
            Incorporate the skill proficiency levels naturally in the CV \
(e.g., "Advanced Python", "Expert React developer", etc.).

            IMPORTANT: Return ONLY the CV content in markdown format. \
Do NOT include any code block markers like ```markdown or ``` in your response.\
        """)

        llm_response = self.llm.invoke(cv_generation_prompt)
        cv_markdown_content = llm_response.content.replace("```markdown", "").replace("```", "").strip()

        if not cv_markdown_content:
            raise ValueError(f"LLM returned empty CV content for {profile.full_name}")

        return cv_markdown_content

    def generate_rfp_markdown_via_llm(self, rfp: RfpRecord) -> str:
        """
        Ask Azure OpenAI to produce a professional RFP document in markdown.

        Transforms a structured RfpRecord into a rich, human-readable
        Request for Proposal — complete with executive summary, technical
        requirements, team profile expectations, and evaluation criteria.
        """
        formatted_requirements_lines: list[str] = []
        for requirement in rfp.requirements:
            certification_suffix = (
                f" (Preferred certifications: {', '.join(requirement.preferred_certifications)})"
                if requirement.preferred_certifications else ""
            )
            mandatory_label = "REQUIRED" if requirement.is_mandatory else "Preferred"
            formatted_requirements_lines.append(
                f"- {mandatory_label}: {requirement.skill_name} "
                f"- {requirement.min_proficiency} level{certification_suffix}"
            )

        joined_requirements = "\n".join(formatted_requirements_lines)
        remote_work_label = "Allowed" if rfp.remote_allowed else "Not allowed"

        rfp_generation_prompt = textwrap.dedent(f"""\
            Create a professional RFP (Request for Proposal) document in markdown format \
with the following details:

            Project: {rfp.title}
            Client: {rfp.client}
            Project Type: {rfp.project_type}
            Description: {rfp.description}
            Duration: {rfp.duration_months} months
            Team Size: {rfp.team_size} people
            Budget Range: {rfp.budget_range}
            Start Date: {rfp.start_date}
            Location: {rfp.location}
            Remote Work: {remote_work_label}

            Technical Requirements:
            {joined_requirements}

            Requirements:
            1. Use proper markdown formatting (headers, lists, emphasis)
            2. Structure as a professional PRD (Product Requirements Document)
            3. Include sections like: Executive Summary, Project Overview, \
Technical Requirements, Expected Team Profile, Timeline, Budget, Proposal Guidelines
            4. Create realistic business context and objectives
            5. Add specific deliverables and milestones
            6. Include detailed descriptions of the expected programmer profiles
            7. Make it sound professional and business-oriented
            8. Add acceptance criteria and evaluation process
            9. Include contact information and proposal submission guidelines

            Focus on creating a comprehensive PRD that clearly outlines what the client needs \
and what kind of development team they're looking for.

            IMPORTANT: Return ONLY the RFP content in markdown format. \
Do NOT include any code block markers like ```markdown or ``` in your response.\
        """)

        llm_response = self.llm.invoke(rfp_generation_prompt)
        rfp_markdown_content = llm_response.content.replace("```markdown", "").replace("```", "").strip()

        if not rfp_markdown_content:
            raise ValueError(f"LLM returned empty content for RFP {rfp.rfp_id}")

        return rfp_markdown_content

    @staticmethod
    def save_markdown_as_pdf(
            markdown_content: str,
            filename: str,
            output_directory: Path,
    ) -> Path:
        """Convert markdown text to a styled PDF and return the file path."""
        output_directory.mkdir(parents=True, exist_ok=True)
        html_body = markdown.markdown(markdown_content)
        pdf_path = output_directory / f"{filename}.pdf"
        HTML(string=html_body).write_pdf(
            str(pdf_path),
            stylesheets=[CSS(string=CV_PDF_STYLESHEET)],
        )
        return pdf_path

    def generate_all_data(
            self,
            num_programmers: int = 10,
            num_projects: int = 20,
            num_rfps: int = 3,
    ) -> dict[str, Any]:
        """
        Generate all artefacts: profiles + CVs, project records, and RFP documents.

        This is the main orchestration method that drives the entire data
        generation pipeline.  It creates programmer profiles, renders their CVs
        as PDF via the LLM, builds project records with skill-based programmer
        assignments, and produces RFP documents — persisting everything as
        JSON and PDF files under the configured output directories.
        """
        if num_programmers <= 0:
            raise ValueError("Number of programmers must be positive")

        programmers_output_dir = Path(self.config["output"]["programmers_dir"])
        rfps_output_dir = Path(self.config["output"]["rfps_dir"])
        projects_output_dir = Path(self.config["output"]["projects_dir"])

        for output_dir in (programmers_output_dir, rfps_output_dir, projects_output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)

        print_section_header(f"PHASE 1: Generating {num_programmers} programmer profiles and CVs")

        profiles = self.generate_programmer_profiles(profiles_count=num_programmers)

        generated_cv_paths: list[Path] = []
        for cv_number, current_profile in enumerate(profiles, start=1):
            print(f"  Generating CV {cv_number}/{num_programmers}: {current_profile.full_name}")
            cv_markdown = self.generate_cv_markdown_via_llm(current_profile)
            sanitized_name = current_profile.full_name.replace(" ", "_").replace(".", "")
            cv_filename = f"cv_{current_profile.profile_id:03d}_{sanitized_name}"
            cv_path = self.save_markdown_as_pdf(cv_markdown, cv_filename, programmers_output_dir)
            generated_cv_paths.append(cv_path)

        print_section_header(f"PHASE 2: Generating {num_projects} project records with programmer assignments")

        project_records = self.generate_project_records(
            projects_count=num_projects,
            programmer_profiles=profiles,
        )

        print_section_header(f"PHASE 3: Generating {num_rfps} RFP records and PDFs")

        rfp_records = self.generate_rfp_records(rfps_count=num_rfps)

        generated_rfp_paths: list[Path] = []
        for rfp_number, current_rfp in enumerate(rfp_records, start=1):
            print(f"  Generating RFP PDF {rfp_number}/{num_rfps}: {current_rfp.title}")
            rfp_markdown = self.generate_rfp_markdown_via_llm(current_rfp)
            sanitized_title = current_rfp.title.replace(" ", "_").replace(".", "").replace("/", "_")
            rfp_filename = f"rfp_{current_rfp.rfp_id}_{sanitized_title}"
            rfp_path = self.save_markdown_as_pdf(rfp_markdown, rfp_filename, rfps_output_dir)
            generated_rfp_paths.append(rfp_path)

        profiles_json_path = programmers_output_dir / "programmer_profiles.json"
        _write_json(profiles_json_path, [profile.to_serializable_dict() for profile in profiles])

        projects_json_path = projects_output_dir / "projects.json"
        _write_json(projects_json_path, [record.to_serializable_dict() for record in project_records])

        rfps_json_path = rfps_output_dir / "rfps.json"
        _write_json(rfps_json_path, [record.to_serializable_dict() for record in rfp_records])

        print(textwrap.dedent(f"""\

            ✅ Generated {len(generated_cv_paths)} CVs in {programmers_output_dir}/
            ✅ Generated {len(generated_rfp_paths)} RFP PDFs in {rfps_output_dir}/
            ✅ Saved {len(profiles)} profiles to {profiles_json_path}
            ✅ Saved {len(project_records)} projects to {projects_json_path}
            ✅ Saved {len(rfp_records)} RFPs to {rfps_json_path}\
        """))

        return {
            "profiles": profiles,
            "projects": project_records,
            "rfps": rfp_records,
            "cv_files": generated_cv_paths,
            "rfp_files": generated_rfp_paths,
            "profiles_file": profiles_json_path,
            "projects_file": projects_json_path,
            "rfps_file": rfps_json_path,
        }


def _write_json(file_path: Path, data: Any) -> None:
    with file_path.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, default=str)


def main() -> None:
    """Generate all data artifacts for the GraphRAG demonstration."""
    generator = GraphRAGDataGenerator()
    generation_params = generator.config["generation"]

    result = generator.generate_all_data(
        num_programmers=generation_params["num_programmers"],
        num_projects=generation_params["num_projects"],
        num_rfps=generation_params["num_rfps"],
    )

    cv_file_listing = "\n".join(f"  - {cv_path}" for cv_path in result["cv_files"])
    rfp_file_listing = "\n".join(f"  - {rfp_path}" for rfp_path in result["rfp_files"])

    print(textwrap.dedent(f"""\

        📄 CV Files:
        {cv_file_listing}

        📋 RFP Files:
        {rfp_file_listing}

        📊 Data Files:
          - {result['profiles_file']}
          - {result['projects_file']}
          - {result['rfps_file']}\
    """))


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
