import hashlib
import pickle

def cache_response(query: str, response: dict, cache_path: str = "./cache/responses.pkl"):
    """
    Stores the response in a cache file to speed up repeated queries.
    """
    cache_key = hashlib.md5(query.encode()).hexdigest()

    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    except (FileNotFoundError, EOFError):
        cache = {}

    cache[cache_key] = response

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


def retrieve_cached_response(query: str, cache_path: str = "./cache/responses.pkl"):
    """
    Retrieves the cached response if available.
    """
    cache_key = hashlib.md5(query.encode()).hexdigest()

    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    except (FileNotFoundError, EOFError):
        return None

    return cache.get(cache_key)
