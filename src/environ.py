import os

import dotenv

dotenv.load_dotenv()

OPENCAGE_API = os.environ["OPENCAGE_API"]
OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
OLLAMA_MODEL = os.environ["OLLAMA_MODEL"]
