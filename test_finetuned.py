"""
Test the fine-tuned model integration
"""

from openai import OpenAI
from src.utils.config import config

# Initialize client
client = OpenAI(api_key=config.models.openai_api_key)

print("=" * 70)
print("🧪 Testing Fine-Tuned Model")
print("=" * 70)
print(f"Model: {config.models.llm_model}\n")

# Test prompt
test_prompt = "Summarize what deep learning is and its main applications in 2-3 sentences."

print(f"Prompt: {test_prompt}\n")
print("Response:")
print("-" * 70)

# Call fine-tuned model
response = client.chat.completions.create(
    model=config.models.llm_model,
    messages=[
        {"role": "system", "content": "You are an expert research assistant specializing in AI and machine learning."},
        {"role": "user", "content": test_prompt}
    ],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)
print("-" * 70)
print("\n✅ Fine-tuned model is working!")