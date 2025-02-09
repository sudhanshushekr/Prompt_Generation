from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextGenerator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set pad token to EOS token (GPT-2 does not have a pad token by default)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_text(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        
        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],  # Explicitly passing attention mask
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id  # Ensuring proper padding behavior
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

if __name__ == '__main__':
    generator = TextGenerator()
    prompt = "sudhanshu"
    print(generator.generate_text(prompt))
