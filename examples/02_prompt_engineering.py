"""
02_prompt_engineering.py - Master the Art of Prompting

Prompt engineering is the skill of crafting effective inputs to get the best outputs from LLMs.
This example teaches you fundamental prompting techniques with practical examples.

Learning Objectives:
- Understand what makes a good prompt
- Learn different prompting techniques
- Practice with real examples
- See the impact of prompt structure on outputs
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class PromptEngineeringDemo:
    def __init__(self):
        print("🎨 Welcome to Prompt Engineering Masterclass!")
        print("=" * 50)
        
        # Load model
        print("Loading model...")
        self.model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Model ready!")
    
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        """Generate text with the given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                num_return_sequences=1
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def demo_clear_vs_vague(self):
        """Demonstrate the importance of clear, specific prompts"""
        print("\n📝 Lesson 1: Clear vs Vague Prompts")
        print("-" * 40)
        
        # Vague prompt
        vague_prompt = "Write about animals"
        print(f"❌ Vague prompt: '{vague_prompt}'")
        vague_result = self.generate_text(vague_prompt, max_length=60)
        print(f"Result: {vague_result}")
        
        print()
        
        # Clear prompt
        clear_prompt = "Write a short paragraph about how elephants use their trunks to communicate with each other"
        print(f"✅ Clear prompt: '{clear_prompt}'")
        clear_result = self.generate_text(clear_prompt, max_length=80)
        print(f"Result: {clear_result}")
    
    def demo_role_playing(self):
        """Show how assigning roles improves responses"""
        print("\n🎭 Lesson 2: Role-Playing Prompts")
        print("-" * 35)
        
        question = "How do I invest my money?"
        
        # Without role
        basic_prompt = f"Question: {question}\nAnswer:"
        print(f"❌ Basic prompt: '{basic_prompt}'")
        basic_result = self.generate_text(basic_prompt, max_length=70)
        print(f"Result: {basic_result}")
        
        print()
        
        # With role
        role_prompt = f"You are a financial advisor. A client asks: '{question}' Provide helpful advice:"
        print(f"✅ Role-based prompt: '{role_prompt}'")
        role_result = self.generate_text(role_prompt, max_length=80)
        print(f"Result: {role_result}")
    
    def demo_examples_in_prompt(self):
        """Show how examples improve output quality"""
        print("\n📚 Lesson 3: Few-Shot Learning (Examples in Prompts)")
        print("-" * 50)
        
        # Without examples
        task = "Classify the sentiment of this text: 'I love this movie!'"
        no_example_prompt = f"{task}\nSentiment:"
        print(f"❌ Without examples: '{no_example_prompt}'")
        no_example_result = self.generate_text(no_example_prompt, max_length=50)
        print(f"Result: {no_example_result}")
        
        print()
        
        # With examples (few-shot)
        few_shot_prompt = """Classify the sentiment of the following texts as Positive, Negative, or Neutral.

Examples:
Text: "This is amazing!" 
Sentiment: Positive

Text: "I hate waiting in lines."
Sentiment: Negative

Text: "The weather is okay."
Sentiment: Neutral

Text: "I love this movie!"
Sentiment:"""
        
        print(f"✅ With examples: '{few_shot_prompt}'")
        few_shot_result = self.generate_text(few_shot_prompt, max_length=60)
        print(f"Result: {few_shot_result}")
    
    def demo_step_by_step(self):
        """Show how breaking tasks into steps helps"""
        print("\n🪜 Lesson 4: Step-by-Step Thinking")
        print("-" * 35)
        
        problem = "If I buy 3 apples for $2 each and 2 oranges for $1.50 each, how much do I spend in total?"
        
        # Direct question
        direct_prompt = f"Question: {problem}\nAnswer:"
        print(f"❌ Direct approach: '{direct_prompt}'")
        direct_result = self.generate_text(direct_prompt, max_length=50)
        print(f"Result: {direct_result}")
        
        print()
        
        # Step-by-step
        step_prompt = f"""Let's solve this step by step:

Question: {problem}

Step 1: Calculate the cost of apples
Step 2: Calculate the cost of oranges  
Step 3: Add them together

Solution:"""
        
        print(f"✅ Step-by-step: '{step_prompt}'")
        step_result = self.generate_text(step_prompt, max_length=80)
        print(f"Result: {step_result}")
    
    def demo_constraints(self):
        """Show how constraints shape outputs"""
        print("\n⚖️ Lesson 5: Using Constraints")
        print("-" * 30)
        
        topic = "artificial intelligence"
        
        # No constraints
        no_constraint_prompt = f"Explain {topic}:"
        print(f"❌ No constraints: '{no_constraint_prompt}'")
        no_constraint_result = self.generate_text(no_constraint_prompt, max_length=60)
        print(f"Result: {no_constraint_result}")
        
        print()
        
        # With constraints
        constrained_prompt = f"Explain {topic} in exactly 2 sentences using simple words that a 10-year-old could understand:"
        print(f"✅ With constraints: '{constrained_prompt}'")
        constrained_result = self.generate_text(constrained_prompt, max_length=70)
        print(f"Result: {constrained_result}")
    
    def interactive_practice(self):
        """Let users practice prompt engineering"""
        print("\n🎯 Practice Time!")
        print("-" * 20)
        print("Now you try! Craft prompts for these tasks:")
        
        tasks = [
            "Get the model to write a haiku about coding",
            "Make the model explain quantum physics like a pirate",
            "Get a recipe for chocolate cake in bullet points",
            "Ask for a story that ends with 'And that's how I learned to fly'"
        ]
        
        for i, task in enumerate(tasks, 1):
            print(f"\n📋 Task {i}: {task}")
            user_prompt = input("Your prompt: ").strip()
            
            if user_prompt:
                print(f"\nYour prompt: '{user_prompt}'")
                result = self.generate_text(user_prompt, max_length=80)
                print(f"Model output: {result}")
                
                feedback = input("\nHow did it do? (good/bad/okay): ").strip().lower()
                if feedback == 'bad':
                    print("💡 Tip: Try being more specific or adding examples!")
                elif feedback == 'good':
                    print("🎉 Great prompting!")
                else:
                    print("👍 Keep practicing!")
    
    def run_all_demos(self):
        """Run all the prompt engineering demonstrations"""
        self.demo_clear_vs_vague()
        self.demo_role_playing()
        self.demo_examples_in_prompt()
        self.demo_step_by_step()
        self.demo_constraints()
        
        print("\n🎓 Prompt Engineering Principles Summary:")
        print("=" * 45)
        print("1. ✨ BE SPECIFIC: Clear, detailed prompts work better")
        print("2. 🎭 USE ROLES: 'You are a...' helps set context")
        print("3. 📚 PROVIDE EXAMPLES: Show the model what you want")
        print("4. 🪜 BREAK IT DOWN: Complex tasks need step-by-step thinking")
        print("5. ⚖️ SET CONSTRAINTS: Length, style, audience constraints help")
        print("6. 🔄 ITERATE: Test and refine your prompts")
        
        # Interactive practice
        practice = input("\nWant to practice? (y/n): ").strip().lower()
        if practice in ['y', 'yes']:
            self.interactive_practice()
        
        print("\n➡️  Next: Try examples/03_model_comparison.py to see different models in action!")

def main():
    demo = PromptEngineeringDemo()
    demo.run_all_demos()

if __name__ == "__main__":
    main()