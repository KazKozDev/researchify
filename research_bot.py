from typing import List, Dict, Optional, Union
import re
from datetime import datetime, timedelta
from vector_store import VectorStore, RAGSystem
import ollama
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryContext:
    """Context for the current query including any relevant filters or parameters."""
    original_query: str
    search_query: Optional[str] = None
    time_range: Optional[tuple] = None
    categories: Optional[List[str]] = None
    max_results: int = 5
    require_citations: bool = True

class ResearchBot:
    def __init__(
        self,
        vector_store: VectorStore,
        rag_system: RAGSystem,
        model_name: str = "gemma2",
        max_context_length: int = 4096
    ):
        """
        Initialize the research bot with vector store and RAG system.
        
        Args:
            vector_store: Initialized VectorStore instance
            rag_system: Initialized RAGSystem instance
            model_name: Name of the LLM model to use
            max_context_length: Maximum context length for the LLM
        """
        self.vector_store = vector_store
        self.rag_system = rag_system
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.current_query = None

    def _format_suggested_query(self, keywords: str) -> str:
        """Format a suggested search query with proper syntax."""
        # Remove search command words and clean input
        clean_keywords = keywords.lower()
        for cmd in ["find", "search", "papers", "about", "on"]:
            clean_keywords = clean_keywords.replace(cmd, "").strip()
            
        # Split into individual terms
        terms = clean_keywords.split()
        if not terms:
            return clean_keywords
            
        # Create query parts
        query_parts = []
        
        # Add title search
        query_parts.append(f'ti:"{clean_keywords}"')
        
        # Add abstract search
        query_parts.append(f'abs:"{clean_keywords}"')
        
        # Combine with OR
        return " OR ".join(query_parts)

    def _search_papers(self, search_query: str, max_results: int = 5) -> List[Dict]:
        """Perform paper search with the given query."""
        try:
            logger.info(f"Executing search with query: {search_query}")
            
            papers = self.rag_system.query(
                query=search_query,
                k=max_results
            )
            
            logger.info(f"Found {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    async def process_query(self, user_input: str) -> str:
        """
        Process a user query and generate an appropriate response.
        
        Args:
            user_input: User's question or command
            
        Returns:
            Generated response or search results
        """
        try:
            logger.info(f"Processing input: {user_input}")
            
            # If input starts with "find" or "search", suggest a query format
            if user_input.lower().startswith(("find", "search")):
                suggested_query = self._format_suggested_query(user_input)
                return (
                    f"To search for papers, please enter a search query. Here's a suggested format:\n\n"
                    f"{suggested_query}\n\n"
                    f"You can modify this query or enter your own search query using operators like:\n"
                    f"- ti: for title search\n"
                    f"- abs: for abstract search\n"
                    f"- OR to combine multiple searches\n"
                    f"Please enter your search query:"
                )
            
            # Treat the input as a search query and execute search
            papers = self._search_papers(user_input)
            
            if not papers:
                return (
                    "No papers found matching your query. "
                    "Please try a different search query."
                )
            
            # Format results
            response = f"Found {len(papers)} papers:\n\n"
            for i, paper in enumerate(papers, 1):
                response += (
                    f"{i}. {paper['title']}\n"
                    f"   Authors: {paper['authors']}\n"
                    f"   Published: {paper['published']}\n"
                    f"   Abstract: {paper['abstract'][:200]}...\n\n"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return (
                "An error occurred while processing your query. "
                "Please try again with a different search query."
            )

    def _generate_llm_response(
        self,
        prompt: str,
        context_docs: List[Dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response using the LLM with context from papers."""
        # Prepare context from papers
        paper_contexts = []
        for doc in context_docs:
            paper_context = (
                f"Title: {doc['title']}\n"
                f"Authors: {doc['authors']}\n"
                f"Abstract: {doc['abstract']}\n"
                f"Published: {doc['published']}\n"
                "---"
            )
            paper_contexts.append(paper_context)
        
        # Combine context and prompt
        full_context = "\n".join(paper_contexts)
        
        if system_prompt:
            final_prompt = f"{system_prompt}\n\nContext:\n{full_context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            final_prompt = f"Based on these papers:\n{full_context}\n\nQuestion: {prompt}\n\nAnswer:"
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=final_prompt
            )
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."

# Example usage
if __name__ == "__main__":
    # Initialize components
    vector_store = VectorStore()
    rag_system = RAGSystem(vector_store)
    research_bot = ResearchBot(vector_store, rag_system)

    # Example async interaction
    async def example_interaction():
        # Test different query scenarios
        queries = [
            "find papers about quantum computing",  # Will get suggestion
            'ti:"quantum computing" OR abs:"quantum algorithms"',  # Direct search
            "find black holes",  # Will get suggestion
            'ti:"black holes" OR abs:"event horizon"'  # Direct search
        ]
        
        for query in queries:
            print(f"\nInput: {query}")
            response = await research_bot.process_query(query)
            print("Response:", response)

    # Run example
    import asyncio
    asyncio.run(example_interaction())