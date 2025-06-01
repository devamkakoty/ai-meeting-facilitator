from typing import Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass
class FrameworkStep:
    step_id: str
    title: str
    description: str
    key_questions: List[str]
    success_criteria: List[str]
    typical_duration_minutes: int
    facilitator_prompts: List[str]


@dataclass
class Framework:
    name: str
    description: str
    steps: List[FrameworkStep]
    total_duration_minutes: int


class FrameworkEngine:
    def __init__(self):
        self.frameworks = {}
        self.current_framework = None
        self.current_step_index = 0
        self._load_default_frameworks()

    def _load_default_frameworks(self):
        """Load predefined frameworks"""

        # 5 Whys Problem Solving Framework
        five_whys_steps = [
            FrameworkStep(
                step_id="problem_definition",
                title="Problem Definition",
                description="Clearly define the problem you're trying to solve",
                key_questions=[
                    "What exactly is the problem?",
                    "When does this problem occur?",
                    "Who is affected by this problem?",
                    "What is the impact of this problem?"
                ],
                success_criteria=[
                    "Problem is clearly stated",
                    "All participants understand the problem",
                    "Problem scope is defined"
                ],
                typical_duration_minutes=10,
                facilitator_prompts=[
                    "Let's make sure we have a clear problem statement",
                    "Can everyone agree on how to define this problem?",
                    "What specific aspects of this problem should we focus on?"
                ]
            ),
            FrameworkStep(
                step_id="first_why",
                title="First Why",
                description="Ask why the problem occurs",
                key_questions=["Why does this problem happen?"],
                success_criteria=["At least one potential cause identified"],
                typical_duration_minutes=8,
                facilitator_prompts=[
                    "What do you think causes this problem?",
                    "Let's dig deeper into the root cause"
                ]
            ),
            FrameworkStep(
                step_id="second_why",
                title="Second Why",
                description="Ask why the first cause occurs",
                key_questions=["Why does that cause happen?"],
                success_criteria=["Deeper cause identified"],
                typical_duration_minutes=8,
                facilitator_prompts=[
                    "Now let's ask why that happens",
                    "What's behind this cause?"
                ]
            ),
            FrameworkStep(
                step_id="third_why",
                title="Third Why",
                description="Continue digging deeper",
                key_questions=["Why does that deeper cause occur?"],
                success_criteria=["Even deeper understanding achieved"],
                typical_duration_minutes=8,
                facilitator_prompts=[
                    "Let's continue drilling down",
                    "What's the underlying reason for this?"
                ]
            ),
            FrameworkStep(
                step_id="fourth_why",
                title="Fourth Why",
                description="Keep going deeper",
                key_questions=["Why does that occur?"],
                success_criteria=["Near root cause identified"],
                typical_duration_minutes=8,
                facilitator_prompts=[
                    "We're getting closer to the root cause",
                    "What's driving this issue?"
                ]
            ),
            FrameworkStep(
                step_id="fifth_why",
                title="Fifth Why",
                description="Reach the root cause",
                key_questions=["Why does that fundamental issue exist?"],
                success_criteria=["Root cause identified"],
                typical_duration_minutes=10,
                facilitator_prompts=[
                    "This should help us identify the root cause",
                    "What's the fundamental issue here?"
                ]
            ),
            FrameworkStep(
                step_id="solution_development",
                title="Solution Development",
                description="Develop solutions to address the root cause",
                key_questions=[
                    "How can we address the root cause?",
                    "What solutions would prevent this problem?",
                    "Which solution is most feasible?"
                ],
                success_criteria=[
                    "Multiple solutions proposed",
                    "Solutions target the root cause",
                    "Implementation plan outlined"
                ],
                typical_duration_minutes=15,
                facilitator_prompts=[
                    "Now that we know the root cause, what can we do about it?",
                    "What solutions would address this fundamental issue?",
                    "How could we implement these solutions?"
                ]
            )
        ]

        five_whys = Framework(
            name="5 Whys",
            description="A problem-solving technique that asks 'why' five times to get to the root cause",
            steps=five_whys_steps,
            total_duration_minutes=67
        )

        self.frameworks["5_whys"] = five_whys

        # Add more frameworks here (DMAIC, Design Thinking, etc.)

    def get_framework(self, name: str) -> Optional[Framework]:
        """Get a framework by name"""
        return self.frameworks.get(name)

    def set_current_framework(self, name: str) -> bool:
        """Set the current active framework"""
        if name in self.frameworks:
            self.current_framework = self.frameworks[name]
            self.current_step_index = 0
            return True
        return False

    def get_current_step(self) -> Optional[FrameworkStep]:
        """Get the current step in the active framework"""
        if not self.current_framework or self.current_step_index >= len(self.current_framework.steps):
            return None
        return self.current_framework.steps[self.current_step_index]

    def move_to_next_step(self) -> bool:
        """Move to the next step in the framework"""
        if not self.current_framework:
            return False
        if self.current_step_index < len(self.current_framework.steps) - 1:
            self.current_step_index += 1
            return True
        return False

    def get_step_context(self) -> Dict:
        """Get context about current step for LLM"""
        current_step = self.get_current_step()
        if not current_step or not self.current_framework:
            return {}

        return {
            "framework_name": self.current_framework.name,
            "framework_description": self.current_framework.description,
            "current_step": {
                "title": current_step.title,
                "description": current_step.description,
                "key_questions": current_step.key_questions,
                "success_criteria": current_step.success_criteria,
                "facilitator_prompts": current_step.facilitator_prompts
            },
            "step_number": self.current_step_index + 1,
            "total_steps": len(self.current_framework.steps),
            "is_final_step": self.current_step_index == len(self.current_framework.steps) - 1
        }

    def is_step_complete_based_on_conversation(self, conversation: str) -> bool:
        """Analyze if current step seems complete based on conversation content"""
        current_step = self.get_current_step()
        if not current_step:
            return False

        # Simple heuristic - check if success criteria keywords appear in conversation
        conversation_lower = conversation.lower()
        criteria_mentions = 0

        for criteria in current_step.success_criteria:
            criteria_words = criteria.lower().split()
            if any(word in conversation_lower for word in criteria_words):
                criteria_mentions += 1

        # If majority of criteria seem to be addressed, consider step complete
        return criteria_mentions >= len(current_step.success_criteria) * 0.6

    def get_available_frameworks(self) -> List[str]:
        """Get list of available framework names"""
        return list(self.frameworks.keys())