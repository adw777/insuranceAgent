from clients.qdrant_client import get_qdrant_client, query_points_with_filter
from clients.openai_client import OpenAIClient
from clients.token_tracker import TokenTracker
import asyncio

async def main():
    # Connect to your Qdrant instance
    client = get_qdrant_client()
    token_tracker = TokenTracker()
    openai_client = OpenAIClient(token_tracker)

    # Example 3: Multiple values for one field (OR condition)
    # results = scroll_filtered_points(
    #     client=client,
    #     collection_name="policy_chunks2",
    #     filters={"pdf_link": ["https://cms.zurichkotak.com/uploads/Benefit_Illustration_Health_Premier_Advantage_Plan_003_8b878d9ed5.pdf", "https://cms.zurichkotak.com/uploads/Health_Maximiser_Benefit_Illsutration_21072025_7c64de289f.pdf", "https://cms.zurichkotak.com/uploads/Health_Maximiser_Prospectus_fed9c8c2e1.pdf"]}
    # )

    async def get_query_vector():
        return await openai_client.generate_embedding("What is the benefit of the Premier Advantage Plan?")

    query_vector = await get_query_vector()

    results = query_points_with_filter(
        client=client,
        collection_name="policy_chunks2",
        query_vector=query_vector,
        filters={"pdf_link": ["https://cms.zurichkotak.com/uploads/Benefit_Illustration_Health_Premier_Advantage_Plan_003_8b878d9ed5.pdf", "https://cms.zurichkotak.com/uploads/Health_Maximiser_Benefit_Illsutration_21072025_7c64de289f.pdf", "https://cms.zurichkotak.com/uploads/Health_Maximiser_Prospectus_fed9c8c2e1.pdf"]}
    )

    """
    "https://cms.zurichkotak.com/uploads/Health_Maximiser_Benefit_Illsutration_21072025_7c64de289f.pdf", "https://cms.zurichkotak.com/uploads/Health_Maximiser_Prospectus_fed9c8c2e1.pdf"
    """

    print(results)

    # # Print results
    # for result in results:
    #     print(f"ID: {result.id}")
    #     print(f"Payload: {result.payload}")
    #     print("---")

# Call the main function properly
if __name__ == "__main__":
    asyncio.run(main())