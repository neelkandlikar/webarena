import openai
import json
import os
from typing import Any, Dict, List
import numpy as np
from browser_env import DetachedPage
from dataclasses import dataclass, asdict
import re



class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, DetachedPage):
            return str(obj)  # or return {"url": obj.url} if you prefer
        return json.JSONEncoder.default(self, obj)

@dataclass
class StateInfo:
    observation: Dict[str, Any]
    info: Dict[str, Any]

def serialize_state(state: StateInfo) -> str:
    state_copy = StateInfo(**asdict(state))
    
    if 'image' in state_copy.observation:
        state_copy.observation['image'] = "<image_data_placeholder>"
    
    max_text_length = 1000
    if 'text' in state_copy.observation:
        state_copy.observation['text'] = state_copy.observation['text'][:max_text_length]
    
    return json.dumps(asdict(state_copy), cls=CustomEncoder, indent=2)

class LLMJudge:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 500):
        """
        Initialize the LLMJudge.
        
        :param model: The OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        :param temperature: Controls randomness in the output. Higher values make the output more random.
        :param max_tokens: The maximum number of tokens to generate in the response.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Ensure the OpenAI API key is set
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def generate(self, prompt: str) -> dict:
        system_message = """
        You are an AI assistant tasked with evaluating actions for a web browsing agent.
        Your entire response must be valid JSON and nothing else. Use the following format:
        {
            "action_1": {
                "score": <int>,
                "reasoning": <string>
            },
            "action_2": {
                "score": <int>,
                "reasoning": <string>
            }
        }
        Ensure all keys are enclosed in double quotes and all string values are properly escaped.
        Do not include any text outside of this JSON structure.
        """
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                content = response.choices[0].message.content.strip()
                
                # Remove any non-JSON content
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
                
                # Try to parse the JSON
                parsed_json = json.loads(content)
                return parsed_json
            
            except json.JSONDecodeError as json_error:
                print(f"Attempt {attempt + 1} failed. Error: {json_error}")
                if attempt == max_attempts - 1:
                    return {
                        "error": "Failed to generate valid JSON after multiple attempts",
                        "raw_content": content
                    }
            
            except Exception as e:
                print(f"Error in generating response: {e}")
                return {
                    "error": "Failed to generate response",
                    "details": str(e)
                }

    def evaluate_actions(self, actions: List[Dict[str, Any]], state_info: Dict[str, Any], intent: str, trajectory: List[Any]) -> List[float]:
        """
        Evaluate actions using the LLM.
        
        :param actions: List of actions to evaluate.
        :param state_info: Current state information.
        :param intent: The task intent.
        :param trajectory: The trajectory of previous actions and states.
        :return: List of scores for each action.
        """

        print("STATE INFO")
        print(state_info)
        serialized_state = serialize_state(StateInfo(**state_info))
        context = f"Task Intent: {intent}\n\n"
        context += f"Current State: {serialized_state}\n\n"
        context += "Previous Actions:\n"
        for prev_action in trajectory[1::2][-5:]:  # Last 5 actions
            context += f"{json.dumps(prev_action, cls=CustomEncoder, indent=2)}\n"
        
        prompt = f"{context}\n\nProposed Actions:\n"
        for i, action in enumerate(actions):
            prompt += f"Action {i+1}:\n{json.dumps(action, cls=CustomEncoder, indent=2)}\n\n"
        
        prompt += "Evaluate each action based on its potential effectiveness in achieving the task intent. "
        prompt += "Provide a score from 0 to 10 for each action, where 10 is the most effective. "
        prompt += "Explain your reasoning for each score. "
        prompt += "Format your response as JSON with keys 'action_1', 'action_2', etc., each containing 'score' and 'reasoning'."

        evaluation = self.generate(prompt)
        
        print("EVALUATION")
        print(evaluation)
        scores = [evaluation[f'action_{i+1}']['score'] for i in range(len(actions))]
        
        # Log the LLM's reasoning
        for i, score in enumerate(scores):
            reasoning = evaluation[f'action_{i+1}']['reasoning']
            print(f"Action {i+1} Score: {score}/10")
            print(f"Reasoning: {reasoning}")
        
        return scores
        

# Example usage
if __name__ == "__main__":
    llm_judge = LLMJudge(model="gpt-4", temperature=0.7)
    
    # Example data
    actions = [
        {"type": "click", "element": "button", "id": "submit"},
        {"type": "type", "element": "input", "id": "search", "text": "OpenAI"},
    ]
    state_info = {"current_url": "https://example.com", "page_title": "Example Page"}
    intent = "Search for information about OpenAI"
    trajectory = [
        {"type": "load", "url": "https://example.com"},
        {"type": "click", "element": "button", "id": "menu"},
    ]
    
    scores = llm_judge.evaluate_actions(actions, state_info, intent, trajectory)
    print(f"Action scores: {scores}")