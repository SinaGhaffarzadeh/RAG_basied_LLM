
'''
This code implement by llamaindex and sentence-transforemr libraries.
Adding codes to control errors (try-except)
'''

# Importing libraries
import torch
from huggingface_hub import whoami, login
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from llama_index.core.node_parser import SimpleNodeParser


# Checking availibity of Cuda on system
print('Cuda is available!', torch.cuda.is_available())  # Should return True
print("The version of Cuda is:",torch.version.cuda)         # Should match something like '12.1'

# # Downloading the model using its ID, which is taken from Huggingface.
# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Main model
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

# # Downloading embedding model using "SentenceTransformer" library
# emb_model = SentenceTransformer('all-MiniLM-L6-v2')
# print("Model downloaded and cached.")

# Loading pdf files from directory
documents = SimpleDirectoryReader("Data").load_data() 

# Parsing and Indexing all data
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)

# Summoning Embedding model and Embedding all indexed data 
embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")
index = VectorStoreIndex(nodes, embed_model=embed_model)

# LLM 
llm = HuggingFaceLLM(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",  # or "cuda" if using GPU
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.7, "do_sample": True},
)

# Logining to API
login(token="...", add_to_git_credential=False) # Add_your_HuggingFace_Token
print(whoami())

# Query system baised on Language model
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("what is motion?")

print(response)