from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from langchain_community.llms import Ollama
from langchain.llms.base import LLM
from pydantic import Field
from config import AZURE_PROJECT_CONN_STRING


def get_azure_chat_client():
    # This function is called only if the user selects the Azure model.
    project = AIProjectClient.from_connection_string(
        conn_str=AZURE_PROJECT_CONN_STRING, credential=DefaultAzureCredential()
    )
    return project.inference.get_chat_completions_client()

class AzureChatLLM(LLM):
    model: str = Field(..., description="The Azure model deployment name")
    chat_client: any = Field(..., description="The Azure Chat client instance")
    temperature: float = Field(1.0)
    frequency_penalty: float = Field(0.5)
    presence_penalty: float = Field(0.5)

    class Config:
        extra = "allow"

    @property
    def _llm_type(self) -> str:
        return "azure_chat_llm"

    def _call(self, prompt: str, stop=None) -> str:
        messages = [{"role": "system", "content": prompt}]
        response = self.chat_client.complete(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        content = response.choices[0].message.content
        return content

    def run(self, prompt: str, stop=None) -> str:
        return self._call(prompt, stop)

    def bind_tools(self, tools):
        object.__setattr__(self, "tools", tools)
        return self

class AzureEnsembleLLM(LLM):
    llms: list = Field(..., description="List of AzureChatLLM models to ensemble")

    class Config:
        extra = "allow"

    @property
    def _llm_type(self) -> str:
        return "azure_ensemble_llm"

    def _call(self, prompt: str, stop=None) -> str:
        responses = [llm._call(prompt, stop=stop) for llm in self.llms]
        combined = "\n".join(responses)
        return combined

    def run(self, prompt: str, stop=None) -> str:
        return self._call(prompt, stop)

    def bind_tools(self, tools):
        object.__setattr__(self, "tools", tools)
        return self

class OllamaWrapper(LLM):
    model: str = Field(..., description="The Ollama model name")
    temperature: float = Field(0.2)

    class Config:
        extra = "allow"

    def __init__(self, model: str, temperature: float):
        super().__init__(model=model, temperature=temperature)
        self._ollama = Ollama(model=model, temperature=temperature)

    @property
    def _llm_type(self) -> str:
        return "ollama_wrapper"

    def _call(self, prompt: str, stop=None) -> str:
        return self._ollama(prompt, stop=stop)

    def run(self, prompt: str, stop=None) -> str:
        return self._call(prompt, stop)

    def bind_tools(self, tools):
        object.__setattr__(self, "tools", tools)
        return self

# def get_llm(model_choice: str):
#     model_choice = model_choice.lower()
#     if model_choice == "azure":
#         # Only initialize the Azure client if needed.
#         azure_chat_client = get_azure_chat_client()
#         azure_llm1 = AzureChatLLM(
#             model="gpt-4o-mini",
#             chat_client=azure_chat_client,
#             temperature=1,
#             frequency_penalty=0.5,
#             presence_penalty=0.5,
#         )
#         azure_llm2 = AzureChatLLM(
#             model="phi-4",
#             chat_client=azure_chat_client,
#             temperature=1,
#             frequency_penalty=0.5,
#             presence_penalty=0.5,
#         )
#         return AzureEnsembleLLM(llms=[azure_llm1, azure_llm2])
#     elif model_choice == "ollama":
#         return OllamaWrapper(model="vicuna", temperature=0.2)
#     elif model_choice == "qwen2.5":
#         return OllamaWrapper(model="qwen2.5", temperature=0.2)
#     else:
#         raise ValueError("Invalid model choice. Please choose 'azure', 'ollama', or 'qwen2.5'.")

# since we have readonlyerror

def get_llm(model_choice: str):
    model_choice = model_choice.lower()
    if model_choice == "ollama":
        return OllamaWrapper(model="vicuna", temperature=0.2)
    elif model_choice == "qwen2.5":
        return OllamaWrapper(model="qwen2.5", temperature=0.2)
    else:
        raise ValueError("Invalid model choice. Please choose 'ollama' or 'qwen2.5'.")