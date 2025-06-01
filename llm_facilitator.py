import json
import google.generativeai as genai
from typing import Dict, Optional, List
from dataclasses import dataclass
from framework_engine import FrameworkEngine
import os


@dataclass
class FacilitatorResponse:
    should_respond: bool
    message: str
    suggested_action: str  # "continue", "next_step", "clarify", "encourage"
    confidence: float
    reasoning: str


class LLMFacilitator:
    def __init__(self, api_key: str, framework_engine: FrameworkEngine):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.framework_engine = framework_engine
        self.conversation_history = []

    def _build_system_prompt(self, framework_context: Dict, meeting_goal: str) -> str:
        """Build the system prompt for the LLM"""

        base_prompt = f"""You are an AI meeting facilitator and coach helping a team work through a structured problem-solving session. Your role is similar to a skilled teacher who guides students to find answers themselves rather than giving direct answers.

MEETING CONTEXT:
- Goal: {meeting_goal}
- Framework: {framework_context.get('framework_name', 'General Discussion')}
- Current Step: {framework_context.get('current_step', {}).get('title', 'Unknown')} ({framework_context.get('step_number', 0)}/{framework_context.get('total_steps', 0)})

CURRENT STEP DETAILS:
- Description: {framework_context.get('current_step', {}).get('description', '')}
- Key Questions: {', '.join(framework_context.get('current_step', {}).get('key_questions', []))}
- Success Criteria: {', '.join(framework_context.get('current_step', {}).get('success_criteria', []))}

FACILITATION GUIDELINES:
1. Act as a guide, not an answer-giver. Ask probing questions, and provide useful solutions when the participants are stuck for a long time.
2. Keep participants focused on the current step and its objectives.6
3. Only respond when you can add meaningful value to the discussion.
4. Be encouraging and supportive while maintaining focus.
5. If the step seems complete, guide them to the next step.
6. Keep responses concise (1-3 sentences max).
7. Use a warm, professional tone.

RESPONSE DECISION CRITERIA:
- Respond if: participants seem stuck, off-topic, need encouragement, or step appears complete
- Don't respond if: discussion is progressing well, participants are actively engaged in the right direction
- Always include your reasoning for responding or not responding

Your response should be in JSON format:
{{
    "should_respond": boolean,
    "message": "your facilitation message (empty if should_respond is false)",
    "suggested_action": "continue|next_step|clarify|encourage|redirect",
    "confidence": float (0.0 to 1.0),
    "reasoning": "brief explanation of your decision"
}}"""

        return base_prompt

    def analyze_conversation(self, recent_conversation: str, framework_context: Dict,
                             meeting_goal: str) -> FacilitatorResponse:
        """Analyze the conversation and determine if/how to respond"""

        system_prompt = self._build_system_prompt(framework_context, meeting_goal)

        conversation_prompt = f"""
RECENT CONVERSATION:
{recent_conversation}

Based on the conversation above and the current framework step, decide whether to facilitate and how. Remember:
- Only respond if you can genuinely help move the discussion forward
- Focus on asking questions that help participants think deeper
- If they're making good progress, let them continue without interruption
- If the step seems complete based on success criteria, suggest moving forward
"""

        try:
            full_prompt = f"{system_prompt}\n\n{conversation_prompt}"
            response = self.model.generate_content(full_prompt)

            # Parse JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]

            response_data = json.loads(response_text)

            return FacilitatorResponse(
                should_respond=response_data.get('should_respond', False),
                message=response_data.get('message', ''),
                suggested_action=response_data.get('suggested_action', 'continue'),
                confidence=response_data.get('confidence', 0.5),
                reasoning=response_data.get('reasoning', '')
            )

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response.text}")
            # Fallback response
            return FacilitatorResponse(
                should_respond=False,
                message="",
                suggested_action="continue",
                confidence=0.0,
                reasoning="Failed to parse LLM response"
            )
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return FacilitatorResponse(
                should_respond=False,
                message="",
                suggested_action="continue",
                confidence=0.0,
                reasoning=f"Error: {str(e)}"
            )

    def process_conversation(self, conversation: str, meeting_goal: str) -> Optional[str]:
        """Main method to process conversation and get facilitation response"""

        framework_context = self.framework_engine.get_step_context()

        # Analyze conversation with LLM
        response = self.analyze_conversation(conversation, framework_context, meeting_goal)

        print(f"[LLM ANALYSIS] Should respond: {response.should_respond}, Action: {response.suggested_action}")
        print(f"[LLM REASONING] {response.reasoning}")

        if not response.should_respond or response.confidence < 0.3:
            return None

        # Handle suggested actions
        if response.suggested_action == "next_step":
            if self.framework_engine.move_to_next_step():
                next_step = self.framework_engine.get_current_step()
                if next_step:
                    step_message = f"Great progress! Let's move to the next step: **{next_step.title}**. {next_step.description}"
                    return f"{response.message}\n\n{step_message}" if response.message else step_message
                else:
                    return f"{response.message}\n\n🎉 Congratulations! You've completed the {self.framework_engine.current_framework.name} framework!"

        return response.message if response.message else None

    def set_meeting_context(self, framework_name: str, meeting_goal: str) -> bool:
        """Set up the meeting context and framework"""
        success = self.framework_engine.set_current_framework(framework_name)
        if success:
            print(f"Framework set to: {framework_name}")
            print(f"Meeting goal: {meeting_goal}")
        return success

    def get_current_step_info(self) -> Dict:
        """Get information about the current step for display"""
        return self.framework_engine.get_step_context()