from .mutation import MutationStrategy
from .llm_loader import LocalLLMLoader
import random

class LLMMutation(MutationStrategy):
    def __init__(self, mutation_rate, model_path, use_quantization=True):
        super().__init__(mutation_rate)
        
        # Load LLM using our safe local loader
        self.llm_loader = LocalLLMLoader(model_path, use_quantization)
        print("LLM Loaded successfully.")

    def mutate(self, chromosome, search_space):
        """
        Mutates the code in the chromosome using the LLM.
        Expected chromosome structure for LLM GA: {'code': "python code string"}
        """
        if random.random() > self.mutation_rate:
             return chromosome

        code = chromosome.get('code')
        if not code:
            return chromosome

        prompt = (
            "You are an expert Neural Network architect. "
            "Improve this PyTorch code to achieve higher accuracy. "
            "You can change layer sizes, add skip connections, or change activation functions. "
            "Keep the class name 'Net'.\n\n"
            f"```python\n{code}\n```\n\n"
            "Provide ONLY the updated Python code. Do not include any explanation or markdown."
        )

        try:
            print("Requesting LLM mutation...")
            new_code = self.llm_loader.generate(prompt)
            
            # Basic cleanup if model outputs markdown
            if "```python" in new_code:
                new_code = new_code.split("```python")[1].split("```")[0].strip()
            elif "```" in new_code:
                new_code = new_code.split("```")[1].split("```")[0].strip()

            if new_code:
                # Basic validation: ensure it has 'class Net' and 'forward'
                if "class Net" in new_code and "def forward" in new_code:
                     # FIX: Preserve other keys (lr, momentum)
                     mutated = chromosome.copy()
                     mutated['code'] = new_code
                     # Reset fitness since code changed
                     mutated['cached_fitness'] = None 
                     return mutated
                else:
                    print("LLM generated invalid code, discarding.")
            else:
                 print("LLM returned empty code.")

        except Exception as e:
            print(f"LLM Mutation failed: {e}")

        return chromosome
