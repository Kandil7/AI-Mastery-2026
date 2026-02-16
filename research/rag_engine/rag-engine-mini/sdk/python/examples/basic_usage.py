"""
Basic Usage Example for RAG Engine Python SDK

This example demonstrates the fundamental operations:
1. Initialize client
2. Ask questions
3. Upload documents
4. Search documents
5. Handle errors
"""

import asyncio
import os
from rag_engine import RAGClient, QueryOptions
from rag_engine.exceptions import AuthenticationError, RateLimitError


async def basic_usage():
    """Demonstrate basic SDK usage."""

    # Get API key from environment
    api_key = os.getenv("RAG_API_KEY", "your-api-key")

    # Initialize client
    # Using context manager ensures proper cleanup
    async with RAGClient(
        api_key=api_key,
        base_url="http://localhost:8000",  # Change to your API URL
        timeout=30.0,
    ) as client:
        print("=" * 60)
        print("RAG Engine Python SDK - Basic Usage Example")
        print("=" * 60)

        # Example 1: Ask a simple question
        print("\n1. Asking a question...")
        try:
            answer = await client.ask(
                question="What is Retrieval-Augmented Generation?",
                k=5,  # Retrieve 5 documents
                use_hybrid=True,
                rerank=True,
            )
            print(f"Answer: {answer.text[:200]}...")
            print(f"Sources: {len(answer.sources)} documents")
            print(f"Retrieval time: {answer.search_ms}ms")
            print(f"LLM time: {answer.llm_ms}ms")
        except Exception as e:
            print(f"Error: {e}")

        # Example 2: Upload a document
        print("\n2. Uploading a document...")
        try:
            # Check if file exists
            file_path = "./sample_document.txt"
            if os.path.exists(file_path):
                doc = await client.upload_document(
                    file_path=file_path,
                    title="Sample Document",
                )
                print(f"Uploaded: {doc.id}")
                print(f"Filename: {doc.filename}")
                print(f"Status: {doc.status}")
                print(f"Size: {doc.size_bytes} bytes")
            else:
                print(f"File not found: {file_path}")
                print("Skipping upload example")
        except Exception as e:
            print(f"Error: {e}")

        # Example 3: Search documents
        print("\n3. Searching documents...")
        try:
            options = QueryOptions(
                limit=10,
                use_hybrid_search=True,
                use_reranking=True,
            )
            results = await client.search_documents(
                query="machine learning",
                options=options,
            )
            print(f"Found {results.total} documents")
            for doc in results.documents[:3]:  # Show first 3
                print(f"  - {doc.filename} ({doc.status})")
        except Exception as e:
            print(f"Error: {e}")

        # Example 4: Get query history
        print("\n4. Getting query history...")
        try:
            history = await client.get_query_history(limit=5)
            print(f"Recent queries: {len(history)}")
            for item in history:
                print(f"  - {item.timestamp}: {item.question[:50]}...")
        except Exception as e:
            print(f"Error: {e}")

        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)


async def error_handling_example():
    """Demonstrate error handling."""

    print("\n\nError Handling Example")
    print("=" * 60)

    # Test with invalid API key
    async with RAGClient(api_key="invalid-key") as client:
        try:
            await client.ask("Test question")
        except AuthenticationError:
            print("✓ AuthenticationError caught (expected)")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")

    # Test rate limiting (if you have a low rate limit)
    async with RAGClient(api_key=os.getenv("RAG_API_KEY", "your-api-key")) as client:
        try:
            # Make many rapid requests
            for i in range(100):
                await client.ask("Test")
        except RateLimitError as e:
            print(f"✓ RateLimitError caught after {i} requests")
            print(f"  Retry after: {e.retry_after} seconds")
        except Exception as e:
            print(f"Note: {e}")


def main():
    """Main entry point."""
    print("RAG Engine Python SDK Examples")
    print("Make sure you have:")
    print("1. RAG Engine API running (e.g., http://localhost:8000)")
    print("2. Valid API key set in RAG_API_KEY environment variable")
    print()

    # Run examples
    asyncio.run(basic_usage())

    # Uncomment to test error handling
    # asyncio.run(error_handling_example())


if __name__ == "__main__":
    main()
