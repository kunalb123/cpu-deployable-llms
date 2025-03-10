'''
This class takes in the path to a gguf file representing an LLM
and allows a user to interact with the LLM running on a CPU.

'''

from llama_cpp import Llama

class CPUModel:

    # pass in the path to the gguf file
    def __init__(self, model_path):
        self.llm = Llama(model_path=model_path)

    # returns the complete response (a dictionary) of the LLM given an input
    # input_text is the string that you wish to input into the LLM
    def get_response(self, input_text):
        return self.llm(input_text)

    # returns the first "choice" that the model outputs. this is just the text
    # that the model outputs without the other information on the response
    # input_text is the string that you wish to input into the LLM
    def get_text_response(self, input_args):
        response = self.get_response(input_args)
        return response['choices'][0]['text']

    # to be implemented
    def evaluate(self): pass