"""
Quick test to verify your LLM setup is working
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_llm():
    print("Testing LLM setup...")
    
    # Load model and tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate text
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=30,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    print("✅ LLM setup is working correctly!")

if __name__ == "__main__":
    test_llm()
