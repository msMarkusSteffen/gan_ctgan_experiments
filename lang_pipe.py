from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
import random

class FhirChain():
    def __init__(self, model, language): 
        self.model = model
        self.language=language      
        self._init_models()    


    def _init_models(self):
        pass
    def build_prompts(self):
        pass

    def run_pipeline(self):
        
        self.evaluation_parser = PydanticOutputParser(pydantic_object=BlogEvaluation)
        format_instructions = self.evaluation_parser.get_format_instructions() # Ruft jetzt BlogEvaluation.model_json_schema() auf



if __name__ == "__main__":
    pass