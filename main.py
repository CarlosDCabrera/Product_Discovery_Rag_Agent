import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def setup_knowledge_base():
    products = [
        "Apple iPhone 14 smartphone with 128GB storage, dual camera system, iOS mobile device electronics",
        "Samsung Galaxy S23 Android smartphone with AMOLED display, fast processor, mobile electronics",
        "Sony WH-1000XM5 wireless noise cancelling headphones, Bluetooth audio device electronics",
        "Apple MacBook Air M2 lightweight laptop with 13-inch display, 8GB RAM, productivity computer",
        "Dell XPS 13 ultrabook laptop with Intel processor, high resolution display, portable computer",
        "Logitech MX Master 3S wireless ergonomic mouse for productivity and office work",
        "Keychron K8 mechanical wireless keyboard with RGB lighting for programmers and gaming",
        "Amazon Kindle Paperwhite e-reader with adjustable light for reading books digitally",
        "Ninja Professional blender for smoothies, frozen drinks, and kitchen food preparation",
        "Instant Pot Duo 7-in-1 electric pressure cooker for quick meals and kitchen cooking",
        "Fitbit Charge 5 fitness tracker for heart rate monitoring and activity tracking",
        "GoPro HERO11 action camera for adventure recording, waterproof outdoor camera",
        "Canon EOS R50 mirrorless camera for photography and video creation",
        "Anker portable power bank 20000mAh for charging phones and mobile devices",
        "Samsung 27 inch 4K UHD monitor for productivity, programming, and gaming",
        "Apple AirPods Pro wireless earbuds with noise cancellation and spatial audio",
        "Nintendo Switch OLED gaming console for handheld and TV gaming",
        "Dyson V8 cordless vacuum cleaner for home cleaning",
        "Philips Hue smart LED light bulb compatible with smart home assistants",
        "JBL Flip 6 portable Bluetooth speaker for music and outdoor audio"
    ]
    # Initialize the embedding model (all-MiniLM-L6-v2 is fast and free)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the vector store from your product list
    vector_store = FAISS.from_texts(products, embeddings)
    return vector_store

vector_db = setup_knowledge_base()

@tool
def search_products(query: str) -> str:
    """Search the product catalog to find items matching the user's request."""
    docs = vector_db.similarity_search(query, k=2)
    return "\n".join([d.page_content for d in docs])

tools = [search_products]

# 3. AGENT CORE
# Explicitly create the model object to avoid the Vertex AI import error
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", # gemini-1.5-flash is faster/cheaper for testing
    temperature=0,
    google_api_key=api_key
)

# Create the agent logic
agent = create_agent (
    model=llm,
    tools=tools,
    system_prompt="You are a helpful product assistant. Use the search tool to find items.",
    debug=True
)


if __name__ == "__main__":
    user_input = "I need something for my daily commute to listen to music and block out noise."
    response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    final_message = response["messages"][-1].content
    print(f"\nFinal Answer: {final_message}")