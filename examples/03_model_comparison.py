"""
03_model_comparison.py - Compare Different Language Models

This example shows you how different language models behave with the same prompts.
You'll learn about model sizes, capabilities, and how to choose the right model for your task.

Learning Objectives:
- Compare different pre-trained models
- Understand model sizes and trade-offs
- See how model choice affects output quality
- Learn when to use which model
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

class ModelComparison:
    def __init__(self):
        print("🔍 LLM Model Comparison Lab")
        print("=" * 40)
        
        # Define models to compare (starting with smaller ones for faster loading)
        self.models_info = {
            "gpt2": {
                "name": "GPT-2 (Small)",
                "size": "124M parameters",
                "description": "Fast, good for learning and simple tasks",
                "model": None,
                "tokenizer": None
            },
            "gpt2-medium": {
                "name": "GPT-2 (Medium)", 
                "size": "355M parameters",
                "description": "Better quality, still relatively fast",
                "model": None,
                "tokenizer": None
            },
            "distilgpt2": {
                "name": "DistilGPT-2",
                "size": "82M parameters",
                "description": "Smaller, faster version of GPT-2",
                "model": None,
                "tokenizer": None
            }
        }
        
        self.loaded_models = {}
    
    def load_model(self, model_key):
        """Load a specific model and tokenizer"""
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        print(f"📦 Loading {self.models_info[model_key]['name']}...")
        start_time = time.time()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_key)
            model = AutoModelForCausalLM.from_pretrained(model_key)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            load_time = time.time() - start_time
            print(f"✅ Loaded in {load_time:.2f} seconds")
            
            self.loaded_models[model_key] = (model, tokenizer)
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Failed to load {model_key}: {e}")
            return None, None
    
    def generate_with_model(self, model, tokenizer, prompt, max_length=50, temperature=0.7):
        """Generate text using a specific model"""
        inputs = tokenizer(prompt, return_tensors="pt")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                num_return_sequences=1
            )
        
        generation_time = time.time() - start_time
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text, generation_time
    
    def compare_models_on_prompt(self, prompt, models_to_test=None):
        """Compare multiple models on the same prompt"""
        if models_to_test is None:
            models_to_test = list(self.models_info.keys())
        
        print(f"\n🎯 Prompt: '{prompt}'")
        print("-" * 60)
        
        for model_key in models_to_test:
            if model_key not in self.models_info:
                continue
                
            model, tokenizer = self.load_model(model_key)
            if model is None:
                continue
            
            info = self.models_info[model_key]
            print(f"\n📝 {info['name']} ({info['size']})")
            
            try:
                generated_text, gen_time = self.generate_with_model(
                    model, tokenizer, prompt, max_length=60
                )
                
                # Extract just the generated part
                completion = generated_text[len(prompt):].strip()
                
                print(f"Output: {completion}")
                print(f"Generation time: {gen_time:.2f}s")
                
            except Exception as e:
                print(f"Error: {e}")
    
    def demonstrate_size_vs_quality(self):
        """Show how model size affects output quality"""
        print("\n🏆 Model Size vs Quality Demonstration")
        print("=" * 45)
        
        test_prompts = [
            "The key to happiness in life is",
            "In the year 2050, technology will",
            "The most important lesson I learned was"
        ]
        
        for prompt in test_prompts:
            self.compare_models_on_prompt(prompt)
            
            print("\n" + "="*60)
    
    def demonstrate_speed_vs_accuracy(self):
        """Show speed vs accuracy trade-offs"""
        print("\n⚡ Speed vs Accuracy Trade-offs")
        print("-" * 35)
        
        prompt = "Write a creative story beginning:"
        models_to_test = ["distilgpt2", "gpt2", "gpt2-medium"]
        
        print("Generating longer text to see quality differences...")
        
        for model_key in models_to_test:
            model, tokenizer = self.load_model(model_key)
            if model is None:
                continue
            
            info = self.models_info[model_key]
            print(f"\n📖 {info['name']}")
            print(f"Description: {info['description']}")
            
            try:
                generated_text, gen_time = self.generate_with_model(
                    model, tokenizer, prompt, max_length=80, temperature=0.8
                )
                
                completion = generated_text[len(prompt):].strip()
                print(f"Story: {completion}")
                print(f"Speed: {gen_time:.2f}s")
                
            except Exception as e:
                print(f"Error: {e}")
    
    def model_selection_guide(self):
        """Provide guidance on choosing models"""
        print("\n📊 Model Selection Guide")
        print("=" * 30)
        
        guide = {
            "For Learning & Experimentation": {
                "Recommended": "gpt2, distilgpt2",
                "Why": "Fast to load, good for understanding concepts"
            },
            "For Creative Writing": {
                "Recommended": "gpt2-medium or larger",
                "Why": "Better coherence and creativity"
            },
            "For Production Apps (Fast Response)": {
                "Recommended": "distilgpt2",
                "Why": "Good balance of speed and quality"
            },
            "For High-Quality Output": {
                "Recommended": "gpt2-large or API models",
                "Why": "Best coherence and knowledge"
            }
        }
        
        for use_case, info in guide.items():
            print(f"\n🎯 {use_case}")
            print(f"   Recommended: {info['Recommended']}")
            print(f"   Why: {info['Why']}")
    
    def interactive_comparison(self):
        """Let users test their own prompts"""
        print("\n🧪 Interactive Model Comparison")
        print("-" * 35)
        
        available_models = [key for key in self.models_info.keys()]
        print(f"Available models: {', '.join(available_models)}")
        
        while True:
            prompt = input("\nEnter your prompt (or 'quit' to exit): ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            # Ask which models to compare
            print(f"\nWhich models to compare? (Enter numbers separated by commas)")
            for i, model_key in enumerate(available_models):
                print(f"{i+1}. {self.models_info[model_key]['name']}")
            
            selection = input("Your choice (e.g., '1,2' or 'all'): ").strip()
            
            if selection.lower() == 'all':
                selected_models = available_models
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in selection.split(',')]
                    selected_models = [available_models[i] for i in indices if 0 <= i < len(available_models)]
                except:
                    print("Invalid selection, using all models")
                    selected_models = available_models
            
            self.compare_models_on_prompt(prompt, selected_models)
    
    def run_full_comparison(self):
        """Run the complete model comparison demonstration"""
        print("This demo will compare different language models on various tasks.")
        print("Note: Larger models may take longer to load!\n")
        
        # Start with basic comparisons
        self.demonstrate_size_vs_quality()
        
        # Show speed vs accuracy
        self.demonstrate_speed_vs_accuracy()
        
        # Provide guidance
        self.model_selection_guide()
        
        print("\n🎓 Key Takeaways:")
        print("- Larger models generally produce better quality text")
        print("- Smaller models are faster and use less memory")
        print("- DistilGPT-2 offers good speed/quality balance")
        print("- Choose your model based on your specific needs")
        
        # Interactive session
        do_interactive = input("\nTry interactive comparison? (y/n): ").strip().lower()
        if do_interactive in ['y', 'yes']:
            self.interactive_comparison()
        
        print("\n➡️  Next: Try examples/04_advanced_prompts.py for advanced prompting techniques!")

def main():
    comparison = ModelComparison()
    comparison.run_full_comparison()

if __name__ == "__main__":
    main()