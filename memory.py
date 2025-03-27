import pickle
from pathlib import Path

MEMORY_FILE = Path("memory.pkl")

def load_memory():
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "rb") as f:
            memory = pickle.load(f)
    else:
        memory = {}
    return memory

def save_memory(memory):
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(memory, f)

def update_memory(memory, query, answer):
    """
    Update the provided memory dictionary with the query and its answer,
    then save the updated memory to file.
    """
    memory[query] = answer
    save_memory(memory)
