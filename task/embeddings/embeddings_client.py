import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


class DialEmbeddingsClient:
    _endpoint: str
    _api_key: str

    def __init__(self, deployment_name: str, api_key: str):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self._endpoint = DIAL_EMBEDDINGS.format(model=deployment_name)
        self._api_key = api_key

    def get_embeddings(self, texts: list[str], dimensions: int = 1536) -> dict[int, list[float]]:
        """
        Generate embeddings for input texts
        
        Args:
            texts: List of text strings to embed
            dimensions: Embedding dimensions (default 1536 for OpenAI models)
            
        Returns:
            Dictionary mapping index to embedding vector
        """
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json"
        }
        
        request_data = {
            "input": texts,
            "dimensions": dimensions
        }

        response = requests.post(
            url=self._endpoint, 
            headers=headers, 
            json=request_data, 
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            embeddings_dict = {}
            
            for item in data.get("data", []):
                index = item.get("index")
                embedding = item.get("embedding")
                embeddings_dict[index] = embedding
            
            return embeddings_dict
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
