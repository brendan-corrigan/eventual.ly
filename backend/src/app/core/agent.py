import supabase
import vecs


class AgentInterface:
    """
    this class is responsible for handling the chatbot interface with the PGVector store
    """

    def __init__(self, supabase_url, supabase_key):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase_client = supabase.Client(supabase_url, supabase_key)
        self.vector_store = vecs.PGVectorStore(self.supabase_client)

    def get_vector(self, key):
        return self.vector_store.get_vector(key)

    def set_vector(self, key, vector):
        return self.vector_store.set_vector(key, vector)

    def delete_vector(self, key):
        return self.vector_store.delete_vector(key)

    def get_all_vectors(self):
        return self.vector_store.get_all_vectors()

    def get_nearest_vectors(self, vector, n):
        return self.vector_store.get_nearest_vectors(vector, n)
