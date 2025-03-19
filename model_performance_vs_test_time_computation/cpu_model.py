'''
This class takes in the path to a gguf file representing an LLM
and allows a user to interact with the LLM running on a CPU.

'''

from llama_cpp import Llama

class CPUModel:

    # pass in the path to the gguf file
    def __init__(self, model_path):
        self.llm = Llama(model_path=model_path, n_gpu_layers=0)

    # returns the complete response (a dictionary) of the LLM given an input
    # args is a dict containing input_text: string that is the input to the LLM
    # as well as other optional model parameters like temperature (float),
    # max_tokens (int), and top_p (float)
    def get_response(self, args: dict):
        return self.llm(
            args['input_text'],
            max_tokens=args.get('max_tokens'),
            temperature=args.get('temperature'),
            top_p=args.get('top_p')
        )

    # returns the first "choice" that the model outputs. this is just the text
    # that the model outputs without the other information on the response
    # args is a dict containing input_text: string that is the input to the LLM
    # as well as other optional model parameters like temperature (float),
    # max_tokens (int), and top_p (float)
    def get_text_response(self, args: dict):
        response = self.get_response(args)
        return response['choices'][0]['text']

    # to be implemented
    def evaluate(self): pass