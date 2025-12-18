from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def _truncate_table(self):
        """Truncate the vectors table to clean up old data"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE vectors")
                conn.commit()
        finally:
            conn.close()

    def _save_chunk(self, document_name: str, text: str, embedding: list[float]):
        """Save a text chunk with its embedding to the database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                embedding_str = str(embedding)
                cur.execute(
                    "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                    (document_name, text, embedding_str)
                )
                conn.commit()
        finally:
            conn.close()

    def process_text_file(
        self, 
        file_path: str, 
        chunk_size: int = 300, 
        overlap: int = 40, 
        dimensions: int = 1536,
        truncate_table: bool = True
    ):
        """
        Process a text file: load, chunk, embed, and store in database
        
        Args:
            file_path: Path to the text file
            chunk_size: Size of text chunks
            overlap: Character overlap between chunks
            dimensions: Embedding dimensions
            truncate_table: Whether to truncate the table before processing
        """
        # Truncate table if requested
        if truncate_table:
            self._truncate_table()
        
        # Load file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Generate chunks
        chunks = chunk_text(content, chunk_size, overlap)
        
        # Generate embeddings for all chunks
        embeddings_dict = self.embeddings_client.get_embeddings(chunks, dimensions)
        
        # Extract document name from file path
        import os
        document_name = os.path.basename(file_path)
        
        # Save each chunk with its embedding
        for idx, chunk in enumerate(chunks):
            embedding = embeddings_dict[idx]
            self._save_chunk(document_name, chunk, embedding)

    def search(
        self, 
        query: str, 
        search_mode: SearchMode = SearchMode.COSINE_DISTANCE,
        top_k: int = 5, 
        min_score: float = 0.5,
        dimensions: int = 1536
    ) -> list[str]:
        """
        Search for relevant text chunks using semantic similarity
        
        Args:
            query: User's search query
            search_mode: Distance metric (cosine or euclidean)
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            dimensions: Embedding dimensions
            
        Returns:
            List of relevant text chunks
        """
        # Generate embedding for the query
        query_embeddings = self.embeddings_client.get_embeddings([query], dimensions)
        query_embedding = query_embeddings[0]
        query_embedding_str = str(query_embedding)
        
        # Choose distance operator based on search mode
        if search_mode == SearchMode.COSINE_DISTANCE:
            distance_op = "<=>"
        else:  # EUCLIDIAN_DISTANCE
            distance_op = "<->"
        
        # Search in database
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query_sql = f"""
                    SELECT text, embedding {distance_op} %s::vector as distance
                    FROM vectors
                    WHERE embedding {distance_op} %s::vector < %s
                    ORDER BY distance
                    LIMIT %s
                """
                cur.execute(query_sql, (query_embedding_str, query_embedding_str, 1 - min_score, top_k))
                results = cur.fetchall()
                return [row['text'] for row in results]
        finally:
            conn.close()

