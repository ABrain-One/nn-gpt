class OnnxCausalLMWrapper:
    """
    Thin wrapper to make ORTModelForCausalLM compatible with ChatBot_Onnx.
    Provides .eval() (no-op) and forwards .generate() and .__call__().
    """
    def __init__(self, ort_model):
        self.ort_model = ort_model
        self.config = ort_model.config
        self.device = 'cuda:0'  # ONNX models always run on GPU in this setup

    def eval(self):
        # ONNX Runtime has no training/eval mode distinction
        return self

    def __call__(self, *args, **kwargs):
        # Forward pass for generation utilities
        return self.ort_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        # Main generation method used by ChatBot
        return self.ort_model.generate(*args, **kwargs)

    def __getattr__(self, name):
        # Forward any other attribute access to the underlying ONNX model
        return getattr(self.ort_model, name)
