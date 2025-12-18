from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


SYSTEM_PROMPT = """You are a helpful RAG-powered assistant that answers questions about microwave oven usage.

You will receive messages in the following structure:
1. RAG Context: Relevant excerpts from the microwave manual
2. User Question: The user's actual question

INSTRUCTIONS:
- Use ONLY the information provided in the RAG Context to answer questions
- If the RAG Context doesn't contain relevant information to answer the question, say "I don't have enough information in the manual to answer that question."
- Do NOT answer questions unrelated to microwave usage or operation
- Do NOT answer questions that are out of scope of the provided context
- Stay focused on the microwave manual content only
- Be concise and helpful in your responses
"""

USER_PROMPT = """RAG Context:
{context}

---

User Question:
{question}
"""


# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres'
}

# Initialize clients
embeddings_client = DialEmbeddingsClient('text-embedding-3-small-1', API_KEY)
chat_client = DialChatCompletionClient('gpt-4o', API_KEY)
text_processor = TextProcessor(embeddings_client, DB_CONFIG)


def process_document():
    """Process the microwave manual and store embeddings in the database"""
    import os
    
    # Get the path to the microwave manual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    manual_path = os.path.join(current_dir, 'embeddings', 'microwave_manual.txt')
    
    print("Processing microwave manual...")
    text_processor.process_text_file(
        file_path=manual_path,
        chunk_size=300,
        overlap=40,
        dimensions=1536,
        truncate_table=True
    )
    print("Document processing completed!\n")


def run_rag_chat():
    """Run the RAG-powered console chat"""
    conversation = Conversation()
    
    # Add system message
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))
    
    print("Microwave Assistant")
    print("=" * 50)
    print("Ask me anything about your microwave oven!")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        # Step 1: Retrieval - Search for relevant context
        relevant_chunks = text_processor.search(
            query=user_input,
            search_mode=SearchMode.COSINE_DISTANCE,
            top_k=5,
            min_score=0.5,
            dimensions=1536
        )
        
        # Step 2: Augmentation - Combine context with user question
        context = "\n\n".join(relevant_chunks) if relevant_chunks else "No relevant context found."
        augmented_prompt = USER_PROMPT.format(context=context, question=user_input)
        
        # Add user message to conversation
        conversation.add_message(Message(Role.USER, augmented_prompt))
        
        # Step 3: Generation - Get LLM response
        try:
            response = chat_client.get_completion(conversation.get_messages())
            conversation.add_message(response)
            
            print(f"\nAssistant: {response.content}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    # Process the document first (comment out after first run if needed)
    process_document()
    
    # Run the chat
    run_rag_chat()
