import dataclasses
from typing import Any, Literal

import mesop as me
from pydantic.dataclasses import dataclass

ContentPart = str | dict[str, Any]


@dataclass
class StateConversation:
    """StateConversation provides mesop state compliant view of a conversation"""

    conversation_id: str = ''
    conversation_name: str = ''
    is_active: bool = True
    message_ids: list[str] = dataclasses.field(default_factory=list)


@dataclass
class StateMessage:
    """StateMessage provides mesop state compliant view of a message"""

    message_id: str = ''
    role: str = ''
    content: list[tuple[ContentPart, str]] = dataclasses.field(
        default_factory=list
    )


@dataclass
class StateTask:
    """StateTask provides mesop state compliant view of task"""

    task_id: str = ''
    session_id: str | None = None
    state: str | None = None
    message: StateMessage = dataclasses.field(default_factory=StateMessage)
    artifacts: list[list[tuple[ContentPart, str]]] = dataclasses.field(
        default_factory=list
    )


@dataclass
class SessionTask:
    """SessionTask organizes tasks based on conversation"""

    session_id: str = ''
    task: StateTask = dataclasses.field(default_factory=StateTask)


@dataclass
class StateEvent:
    """StateEvent provides mesop state compliant view of event"""

    conversation_id: str = ''
    actor: str = ''
    role: str = ''
    id: str = ''
    content: list[tuple[ContentPart, str]] = dataclasses.field(
        default_factory=list
    )


@me.stateclass
class AppState:
    """Mesop Application State"""

    sidenav_open: bool = False
    theme_mode: Literal['system', 'light', 'dark'] = 'system'

    current_conversation_id: str = ''
    conversations: list[StateConversation]
    messages: list[StateMessage]
    task_list: list[SessionTask] = dataclasses.field(default_factory=list)
    background_tasks: dict[str, str] = dataclasses.field(default_factory=dict)
    message_aliases: dict[str, str] = dataclasses.field(default_factory=dict)
    completed_forms: dict[str, dict[str, Any] | None] = dataclasses.field(
        default_factory=dict
    )
    form_responses: dict[str, str] = dataclasses.field(default_factory=dict)
    polling_interval: int = 1

    api_key: str = ''
    uses_vertex_ai: bool = False
    api_key_dialog_open: bool = False


@me.stateclass
class SettingsState:
    """Settings State"""

    output_mime_types: list[str] = dataclasses.field(
        default_factory=lambda: [
            'image/*',
            'text/plain',
        ]
    )
