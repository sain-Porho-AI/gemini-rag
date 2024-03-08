# Multimodel RAG with Gemini Pro and LangChain 
from dotenv import load_dotenv
import os
load_dotenv()
import os
import getpass
import requests
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
import matplotlib.pyplot as plt

GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY')

def get_image(url, filename):
    content = requests.get(url).content
    with open(f'/content/{filename}.png', 'wb') as f:
        f.write(content)
        image = Image.open(f"/content/{filename}.png")
        image.show()
    return image
# for simple prompt base qury
# llm = ChatGoogleGenerativeAI(model="gemini-pro")
# result = llm.invoke("Write a ballad about Gemini Pro in around 3 sentences.")
# print(result.content)


# for simple question and ansewer 
# model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
# print(model([
#   SystemMessage(content="Answer only yes or no."),
#   HumanMessage(content="Is Tomato a fruit?"),
#   ]).content)


# loading image 

# image_path = "nike-sneakers-hero.png"
# image = Image.open(image_path)
# img.show()


# Now, let's use this call Gemini Pro Vision model and 
# ask it to tell us bit about this particular image.
# For this, you simply need to change the model name to gemini-pro-vision.

# llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
# message = HumanMessage(
# content=[
#   {
#     "type": "text",
#     "text": "What's in this image? provide full detail as possible.",
#   }, # You can optionally provide text parts
#   {"type": "image_url", "image_url": image},
# ])
# print(
# llm.invoke([message]).content
# )

def load_image(image_path):
  try:
    image = Image.open(image_path)
    return image
  except FileNotFoundError:
    print(f"Error: Image file '{image_path}' not found.")
    return None

image_path = "nike-sneakers-hero.png"
image = load_image(image_path)

# llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
# message = HumanMessage(
#   content=[
#     {
#       "type": "text",
#       "text": "Describe the image content in detail.",
#     },
#     {"type": "image_url", "image_url": image},
#   ]
# )
# print(llm.invoke([message]).content)

#RAG using Multimodal
store_information = """

Nike Air Max Plus sneakers. They feature a brown upper with a black Nike Swoosh logo on the side and a visible Air Max unit in the heel. The sole is white.
Here are some more details about the Nike Air Max Plus:
Style: TN
Release date: January 1, 2017
Style code: 852630-300
Original retail price: $150 USD
The Air Max Plus, also known as the TN, is a popular Nike running shoe that was first released in 1998. It is known for its unique design, which includes a gradient upper, visible Air Max units, and a wavy outsole. The TN has been a popular shoe among sneakerheads and casual wearers alike for over two decades.
It features a brown upper with a black Swoosh logo and a white sole. The shoe is currently available for resale on the StockX marketplace for an average price of around $150 USD.
Nike Air Max Plus Store Location: "Kings Way, Kathmandu, Nepal".


Nike Dunk Low Retro sneakers.
Here are some more details about the Nike Dunk Low Retro:
Style: Low-top
Release date: October 31, 2020
Style code: DD1391-100
Original retail price: $100 USD
The Dunk Low Retro is a popular Nike skateboarding shoe that was first released in 1985. It is a classic silhouette that has been released in a wide variety of colors and materials over the years.
The shoes in the image are currently available for resale on the StockX marketplace for an average price of around $200 USD.
Nike Dunk Low Retro sneakers Store Location: "Patan, Lalitpur, Nepal".


Nike slide/sandal.
They appear to be the Nike Slide/Sandal, which is a unisex slide/sandal.
Here are some of the features:
Soft, one-piece upper: The upper is made of a soft, synthetic material that provides comfort and support.
Phylon midsole: The midsole is made of Phylon, which provides cushioning and support.
Rubber outsole: The outsole is made of rubber for traction and durability.
Swoosh logo: The Nike Swoosh logo is on the strap of the sandal.
Available in a variety of colors: The Nike Benassi Solarsoft Sandal is available in a variety of colors, including black, white, and beige.
Nike off courte slides store location: "Hyderabad, Sindh,  Pakistan".

"""

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = FAISS.from_texts(
    [store_information], embedding=embeddings
)
retriever = vectorstore.as_retriever()

llm_text = ChatGoogleGenerativeAI(model="gemini-pro")
template = """
```
{context}
```

{information}


Provide brief information and store location.
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "information": RunnablePassthrough()}
    | prompt
    | llm_text
    | StrOutputParser()
)

llm_vision = ChatGoogleGenerativeAI(model="gemini-pro-vision", temperature=0.0)
full_chain = (
    RunnablePassthrough() | llm_vision | StrOutputParser() | rag_chain
)



url_1 = "nike-sneakers-hero.png"
image = load_image(url_1)
# plt.imshow(image)
# plt.show()

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Provide information on Brand and model of given sneaker.",
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": image},
    ]
)


result = full_chain.invoke([message])
print(result)

# this code is for rag and use it for other models 
#hi ishaque
# add somthing new here