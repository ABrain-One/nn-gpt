from transformers import PreTrainedTokenizerBase, PreTrainedModel


class ChatBot:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer

    def chat(self, prompt: str, max_len=None, max_words=None, engineer_prompt=True) -> str:

        # TODO: Uncomment and correct all following code if necessary.


        # if engineer_prompt:
        #     prompt += " Use PyTorch for the implementation. Keep the code short."
        #     prompt += ' Name the main class of the model "Net".'
        #     prompt += ' The model code must include default parameters for initialization in the constructor.'
        #     prompt += " Provide only the code. Don't provide any explanation. "
        #     prompt += "Remove any text from this reply. Don't include comments in the code."
        # prompt = [{"role": "user", "content": prompt}]
        # prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
        # input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        # input_ids = input_ids.to('cuda')
        #
        # if max_len is None and max_words is None:
        #     output = self.model.generate(input_ids, max_length=2048, num_beams=4, no_repeat_ngram_size=2)
        # elif max_words is not None:
        #     output = self.model.generate(input_ids, max_new_tokens=max_words, num_beams=4, no_repeat_ngram_size=2)
        # elif max_len is not None:
        #     output = self.model.generate(input_ids, max_length=max_len, num_beams=4, no_repeat_ngram_size=2)
        # response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # response = str(response)
        # print(response)
        #
        # out = response.split("[/INST] ")[1]

        # TODO: Replace the following code with a more flexible solution, eliminating the hard-coded input_text value.
        # Start: inserted code
        input_text = "#Write PyTorch code for an efficient classification model that includes self-attention blocks. Use PyTorch for the implementation. Keep the code short. Name the main class of the model \"Net\". The model code must include default parameters for initialization in the constructor. Provide only the code. Don't provide any explanation. Remove any text from this reply. Don't include comments in the code."
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=1024)
        out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # End: inserted code

        good_code = ''
        in_code = False
        for idx, line in enumerate(out.splitlines()):
            if "```" in str(line) and not in_code:
                in_code = True
                continue
            if "```" in str(line) and in_code:
                in_code = False
            if in_code:
                good_code += str(line) + "\n"

        return good_code if good_code != '' else out
