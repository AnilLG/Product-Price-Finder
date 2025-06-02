import streamlit as st
from pydantic import BaseModel, Field
from typing import Optional
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load Groq API key
load_dotenv()

# 1. Define the Product entites or characerstics 
class Product(BaseModel):
    
    product_id: Optional[str] = Field(default=None, description="Unique identifier for the product")
    product_name: Optional[str] = Field(default=None, description="Name of the product")
    product_desc: Optional [str] = Field(default=None, description="Product description")
    product_price: Optional[float] = Field(default=None, ge=0, description="It represents the price of the product")
    category: Optional[str] = Field(default=None, description="It spcified the category of the product")
    rating: Optional[int] = Field(default=0.0, ge=0, le=5, description="Rating of the product in between 0 to 5")

# 2. Build the prompt
system_prompt = """
You are a helpful assistant with deep domain knowledge in product analysis and pricing.When the user gives about any product details.
Provide the below details of the given product:
1. **Product Name**
2. **Product Price** /Please added "$" prefix to the product price.

Note: Return valid and structured information only
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",(system_prompt)),
        ("human",f"{input}")
    ]
)

# building the streamlit UI

st.set_page_config(page_title="🛍️ Prduct Price Finder", page_icon="🛒")
st.title("🛍️ Product Price Finder Bot")
st.markdown("Welcome to the **Price Finder App** with the help of **Google Gemini and LangChain**")

col1, col2 = st.columns(2)

# 3. Choose LLM model
with col1:
    # st.markdown("Model")
    # available_models = [
    #     "gemini-2.0-flash",
    #     "gemini-2.5-pro-preview-05-06",
    #     "gemini-2.0-flash-lite",
    #     "gemini-1.5-pro"
    # ]
    available_models = [
        "deepseek-r1-distill-llama-70b",
        "qwen-qwq-32b",
        "llama-3.1-8b-instant",
        "groq-llama-65b-v1",
        "groq-llama-70b-v2",
        "groq-qwen-14b",
        "groq-llama-2-70b-chat",
        "groq-llama-13b-chat",
    ]

    model_choice = st.selectbox("**Choose Model**", options=available_models)

with col2:
    # st.markdown("Product description")
    product_input = st.text_area("**Enter product description:**",
        placeholder = ("Let me know the brief description of the product")
    )

# 4. Execution
if st.button("Get Details"):
    if model_choice and product_input:
        with st.spinner("Thinking..."):
            # model = ChatGoogleGenerativeAI(model=model_choice)
            model = ChatGroq(model=model_choice)
            structured_output = model.with_structured_output(Product)

            # Create a chain
            chain = prompt | structured_output

            try:
                result = chain.invoke({"input":product_input})
                
                st.success("Execution Successfully Completed")
                st.markdown(f"### Result:")
                st.write(f"**Product Name:** {result.product_name}")
                st.write(f"**Price:** ${result.product_price}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning(f"⚠️ Please choose the model and give us the product description as well.")
