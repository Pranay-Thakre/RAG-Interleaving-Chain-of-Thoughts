import re
from typing import List, Tuple

class MockGenerator:
    """
    Local deterministic LLM generator for testing IRCoT reasoning loops,
    demo execution, and offline verification without external API keys or server setup.
    """
    def __init__(self, engine="mock", temperature=0, max_tokens=300, **kwargs):
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_text_sequence(self, input_text: str) -> List[Tuple[str, float]]:
        """
        Parses input prompt (containing questions, contexts, or previous steps)
        and generates appropriate next step, query, or final answer.
        """
        prompt_lower = input_text.lower()
        
        # Check if the prompt asks for final QA answer
        if "answer:" in prompt_lower or "the answer is" in prompt_lower or "final answer" in prompt_lower:
            if "scott derrickson" in prompt_lower and "ed wood" in prompt_lower:
                return [("yes", 0.0)]
            elif "arthur's magazine" in prompt_lower and "first for women" in prompt_lower:
                return [("Arthur's Magazine", 0.0)]
            elif "dark knight" in prompt_lower and "christopher nolan" in prompt_lower:
                return [("Christina Nolan", 0.0)]
            elif "thriller" in prompt_lower and "michael jackson" in prompt_lower:
                return [("Gary, Indiana", 0.0)]
            elif "harvard" in prompt_lower and "yale" in prompt_lower:
                return [("65", 0.0)]
            else:
                return [("yes", 0.0)]

        # Check if prompt requires a search query generation step
        if "query:" in prompt_lower or "search:" in prompt_lower or "follow-up question" in prompt_lower:
            if "scott derrickson" in prompt_lower and "ed wood" not in prompt_lower:
                return [("Ed Wood nationality", 0.0)]
            elif "dark knight" in prompt_lower and "christopher nolan" not in prompt_lower:
                return [("Christopher Nolan mother", 0.0)]
            elif "thriller" in prompt_lower and "michael jackson" not in prompt_lower:
                return [("Michael Jackson birthplace", 0.0)]
            else:
                return [("Scott Derrickson nationality", 0.0)]

        # Default multi-step CoT reasoning output
        if "scott derrickson" in prompt_lower:
            response = "Scott Derrickson is an American film director. Ed Wood was also an American filmmaker. Therefore, both Scott Derrickson and Ed Wood are of the same nationality. So the answer is yes."
        elif "arthur's magazine" in prompt_lower:
            response = "Arthur's Magazine was founded in 1844. First for Women was founded in 1989. Thus, Arthur's Magazine was founded first. So the answer is Arthur's Magazine."
        elif "dark knight" in prompt_lower:
            response = "The Dark Knight was directed by Christopher Nolan. Christopher Nolan's mother is Christina Nolan. So the answer is Christina Nolan."
        else:
            response = "Based on the provided information, Scott Derrickson and Ed Wood are both American filmmakers. So the answer is yes."

        return [(response, 0.0)]
