#!/usr/bin/env python3
"""
setup_environment.py - Easy Setup for LLM Learning

This script helps you set up your environment for learning LLMs.
It checks dependencies, downloads models, and verifies everything works.

Run this first before trying other examples!
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Good!")
    return True

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Check if requirements.txt exists
        req_file = Path("requirements.txt")
        if not req_file.exists():
            print("❌ requirements.txt not found!")
            return False
        
        # Install basic dependencies first
        basic_deps = ["torch", "transformers", "numpy"]
        
        for dep in basic_deps:
            print(f"Installing {dep}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to install {dep}")
                print(result.stderr)
                return False
        
        print("✅ Basic dependencies installed!")
        print("\n💡 To install all dependencies, run: pip install -r requirements.txt")
        return True
        
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def test_basic_functionality():
    """Test if basic LLM functionality works"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from transformers import AutoTokenizer
        
        # Test tokenizer loading
        print("Loading GPT-2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Test tokenization
        test_text = "Hello, world!"
        tokens = tokenizer.tokenize(test_text)
        
        print(f"✅ Tokenization works! '{test_text}' -> {tokens}")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: pip install transformers torch")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_test_script():
    """Create a simple test script"""
    print("\n📝 Creating test script...")
    
    test_script = '''"""
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
'''
    
    try:
        with open("test_setup.py", "w") as f:
            f.write(test_script)
        print("✅ Created test_setup.py")
        return True
    except Exception as e:
        print(f"❌ Failed to create test script: {e}")
        return False

def show_next_steps():
    """Show what to do next"""
    print("\n🎯 Next Steps:")
    print("=" * 20)
    print("1. Run the test script: python test_setup.py")
    print("2. If it works, try: python examples/01_first_llm.py")
    print("3. Follow the examples in order:")
    print("   - examples/01_first_llm.py")
    print("   - examples/02_prompt_engineering.py") 
    print("   - examples/03_model_comparison.py")
    print("4. Read the documentation in docs/")
    print("5. Have fun learning! 🚀")

def check_disk_space():
    """Check if there's enough disk space for models"""
    print("\n💾 Checking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (1024**3)
        
        if free_gb < 2:
            print(f"⚠️  Low disk space: {free_gb}GB free")
            print("   Models can be 1-2GB each. Consider freeing space.")
        else:
            print(f"✅ Disk space OK: {free_gb}GB free")
        
        return free_gb >= 1
        
    except Exception as e:
        print(f"❌ Could not check disk space: {e}")
        return True  # Assume it's OK

def main():
    """Main setup function"""
    print("🚀 LLM Learning Environment Setup")
    print("=" * 35)
    print("This script will help you set up everything needed to learn LLMs!")
    
    all_good = True
    
    # Check Python version
    if not check_python_version():
        all_good = False
    
    # Check disk space
    if not check_disk_space():
        all_good = False
    
    # Install dependencies
    if not install_dependencies():
        all_good = False
    
    # Test functionality
    if not test_basic_functionality():
        all_good = False
    
    # Create test script
    if not create_test_script():
        all_good = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Setup completed successfully!")
        show_next_steps()
    else:
        print("❌ Setup had some issues.")
        print("💡 Common solutions:")
        print("   - Update Python to 3.8+")
        print("   - Install pip: python -m ensurepip --upgrade")
        print("   - Check internet connection")
        print("   - Free up disk space")
    
    print("\n📚 For help, check:")
    print("   - README.md for detailed instructions")
    print("   - docs/ folder for documentation")
    print("   - GitHub issues for troubleshooting")

if __name__ == "__main__":
    main()