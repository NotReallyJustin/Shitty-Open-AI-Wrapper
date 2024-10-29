import openai

class OpenAIWrapper:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the OpenAI API wrapper with an API key and model.
        
        :param api_key: Your OpenAI API key as a string.
        :param model: The OpenAI model to use, default is "gpt-4".
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = self.api_key

    def generate_text(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7):
        """
        Generate text using the specified model.

        :param prompt: The prompt to send to the model.
        :param max_tokens: The maximum number of tokens in the output.
        :param temperature: Sampling temperature. Higher values mean more risk-taking.
        :return: The generated text.
        """
        try:
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error generating text: {e}")
            return None

    def chat(self, messages: list, max_tokens: int = 150, temperature: float = 0.7):
        """
        Engage in a chat with a model like gpt-4 or gpt-3.5-turbo.
        
        :param messages: List of message dictionaries, e.g., [{"role": "user", "content": "Hi!"}].
        :param max_tokens: Maximum number of tokens for the response.
        :param temperature: Sampling temperature.
        :return: Model's response content.
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            print(f"Error in chat: {e}")
            return None
