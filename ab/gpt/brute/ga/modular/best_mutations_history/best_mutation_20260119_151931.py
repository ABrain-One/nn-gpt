# Best Fitness: 0.42132
# Action: mutate_optimizer
# Date: 2026-01-19 15:27:16.932468

'''
 Prompt Used:
### Task: You are an expert. Rewrite the 'self.optimizer' definition in 'train_setup'.
GOAL: Change optimizer (Adam, RMSprop) or parameters (momentum, decay).
RETURN: Only the 'self.optimizer = ...' block.
### Context (Original Code):
```python
self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.01), 
            momentum=prm.get('momentum', 0.9)
        )
```
### Instructions:
1. Return ONLY the replacement code.
2. Do NOT explain.
3. Maintain indentation.
'''

# Generated Code:
self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.01), 
            momentum=prm.get('momentum', 0.9)
        )
