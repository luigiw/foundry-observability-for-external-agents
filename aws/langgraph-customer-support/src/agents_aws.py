"""AWS-specific LLM provider and agent implementation."""
import sys
from pathlib import Path

# Add shared package to path
shared_path = Path(__file__).parent.parent.parent.parent / "shared"
sys.path.insert(0, str(shared_path))

from langchain_aws import ChatBedrock
from customer_support_agents.agents import CustomerSupportAgents, BaseLLMProvider


class BedrockLLMProvider(BaseLLMProvider):
    """AWS Bedrock LLM provider."""
    
    def __init__(self, model_id: str, temperature: float):
        self.llm = ChatBedrock(
            model_id=model_id,
            model_kwargs={"temperature": temperature, "max_tokens": 1024},
            region_name="us-east-1",
        )
    
    def invoke(self, messages: list, **kwargs):
        """Invoke the Bedrock LLM."""
        return self.llm.invoke(messages, **kwargs)


def create_bedrock_llm(model_name: str, temperature: float) -> BedrockLLMProvider:
    """
    Create a Bedrock LLM instance.
    
    Args:
        model_name: "haiku" or "sonnet"
        temperature: Temperature setting
        
    Returns:
        BedrockLLMProvider instance
    """
    model_map = {
        "haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    }
    model_id = model_map.get(model_name, model_map["sonnet"])
    return BedrockLLMProvider(model_id, temperature)


def get_agents() -> CustomerSupportAgents:
    """Get customer support agents configured for AWS Bedrock."""
    return CustomerSupportAgents(llm_factory=create_bedrock_llm)
