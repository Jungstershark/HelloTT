from pathlib import Path

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from utils.llm import get_completion
from utils.tokens import num_tokens_from_string

from PIL import Image
import os
from io import BytesIO
import base64
from dotenv import load_dotenv
import tempfile

from openai import AzureOpenAI
load_dotenv() 


#import modular functions from script
from scripts.feedback import feedback
from scripts.contact import contact

#code taken from gpt4 vision API documentation lol
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = 'gpt-4-vision'
api_version = '2023-05-15' 

client = AzureOpenAI(
    api_key=api_key,  
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)

PROMPT = """
            The output SHOULD BE CONCISE
            1. IF there is no error in the image, output a CONCISE description of the image
            1. ELSE, identify at least one possible error in the image.
            2. If there are any error messages in red, output the error code accurately 
            """


# Function to encode a local image into data URL 
def image_to_data_url(image):
    # Guess the MIME type of the image based on the file format
    mime_type = Image.MIME[image.format]
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Convert PIL image to BytesIO buffer and then to base64
    buffered = BytesIO()
    image.save(buffered, format=image.format)
    base64_encoded_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def get_responses(data_urls,prompt):
    responses_messages = []  # List to hold the messages data from each response
    for data_url in data_urls:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]}
            ],
            max_tokens=2000
        )
        if response.choices:
            message_content = response.choices[0].message.content
            responses_messages.append(message_content)
    return responses_messages






HFM_FILE = Path("./data/HFM.pdf")
FAISS_INDEX_DIR = "./data/faiss_index"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

QNA_TEMPLATE = """
Use this file to answer the requirements and the question.
{context}

REQUIREMENTS:

- Ensure all answers are clear, concise, and easy to interpret.
- Provide steps in point form, directing users how to solve the issue on the platform
- If an answer cannot be found within the file, explicitly state that the answer is not available.
- Restrict answers to information contained within the file.
- If there is no answer within the file, provide a polite response indicating so.
- If the user has tried the previous steps and it still does not work provide the following answer: 
    Ok! No problem - lets link this up with the team. To do this you can provide a snip shot of your error msg on the side tab and click on the Done tallking to chatbot button

{sys_message}

QUESTION: {question}

ANSWER:
"""

SYSTEM_MESSAGE = "You are a helpful assistant. Behave like I am a new user to this system."

@st.cache_resource
def load_embeddings():
    # equivalent to SentenceTransformerEmbeddings(model_name="all-  LM-L6-v2")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()
vector_store = None
file_raw_text = ""

# # Define a custom markdown function with a purple color
# def colored_markdown(text, color="purple"):
#     return f"<h1 style='color:{color};'>{text}</h1>"

# # Use the custom function to display the title
# st.write(colored_markdown("Hello TT!:"))
st.title("🤖 Hello TT")

HFM_FILE = Path("./data/HFM.pdf")
HFM_FILE_ACTUAL = Path("./data/HFMGuide.pdf")
FAISS_INDEX_DIR = "./data/faiss_index"

# if not HFM_FILE.exists():
#     st.stop()

# if HFM_FILE.exists():
#     file_raw_text = ""
#     with tempfile.NamedTemporaryFile(delete=False) as f:
#         f.write(HFM_FILE.read_bytes())
#         documents = PyMuPDFLoader(f.name).load()
#         file_raw_text = "".join(page.page_content for page in documents)

# n_token = num_tokens_from_string(file_raw_text)

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
# )
# chunked_docs = text_splitter.split_documents(documents)
# vector_store = FAISS.from_documents(chunked_docs, embeddings)
# vector_store.save_local(
#     folder_path=FAISS_INDEX_DIR, index_name=HFM_FILE.name
# )


with st.sidebar:
    # Check if HFM.pdf exists
    if not HFM_FILE_ACTUAL.exists():
        st.error("HFM.pdf file not found in the same directory!")
    else:
        #enabling download of HFM_FILE
        with open(HFM_FILE_ACTUAL, "rb") as pdf_file:
            st.download_button(label="Download HFM User Guide here",
                               data=pdf_file,
                               file_name=HFM_FILE_ACTUAL.name,
                               mime="application/octet-stream")
        # Load FAISS index if HFM.pdf exists
        index_file = Path(f"{FAISS_INDEX_DIR}/{HFM_FILE.name}.faiss")
        if index_file.exists():
            vector_store = FAISS.load_local(
                folder_path=FAISS_INDEX_DIR,
                embeddings=embeddings,
                index_name=HFM_FILE.name,
            )

    screenshots = st.sidebar.file_uploader(
                    "Upload screenshots of your issue (optional):", 
                    accept_multiple_files=True, 
                    type=["png", "jpg", "jpeg"],
                    key='contact_screenshots')
    
    # finish button to end convo
    if st.button("Unable to solve your problem? \nClick here"):
        st.session_state.finished = True
        st.experimental_rerun()


    

def send_message():
    user_input = st.session_state.user_input
    image_input = st.session_state.chat_screenshots
    
    data_urls = []
    if image_input:
        for uploaded_file in image_input:
            image = Image.open(uploaded_file)
            data_url = image_to_data_url(image)
            data_urls.append(data_url)


    if user_input and data_urls:
        st.session_state.conversation.append({"role": "user", "content": user_input, "image":data_urls})
        
        responses = get_responses(data_urls, PROMPT) 
        full_input = f"""
                    {responses}
                    {user_input}
                    """

        context = ""
        docs_and_scores = vector_store.similarity_search_with_score(
            #set k=top_k if k slider is uncommented
            user_input, k=3
        )
        if docs_and_scores:
            for doc, score in docs_and_scores:
                context += f"\n{doc.page_content}"
            updated_input = QNA_TEMPLATE.format(context=context, sys_message=SYSTEM_MESSAGE, question=full_input)
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": updated_input},
            ]
            response = get_completion(messages)
            reply = response.choices[0].message.content
            chatbot_answer = reply.replace("$", r"\$")


    # proceed only if there is input!
    elif user_input:
        #RAG stuff here
        st.session_state.conversation.append({"role": "user", "content": user_input, "image":None})
        context = ""
        docs_and_scores = vector_store.similarity_search_with_score(
            #set k=top_k if k slider is uncommented
            user_input, k=3
        )
        if docs_and_scores:
            for doc, score in docs_and_scores:
                context += f"\n{doc.page_content}"
            updated_input = QNA_TEMPLATE.format(context=context, sys_message=SYSTEM_MESSAGE, question=user_input)
            
            if len(st.session_state.conversation) > 1:
                last_two_dicts = [{"role": d["role"], "content": d["content"]} for d in st.session_state.conversation[-2:]]
            
            messages = last_two_dicts + [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": updated_input},
            ]
            response = get_completion(messages)
            reply = response.choices[0].message.content
            chatbot_answer = reply.replace("$", r"\$")
            

    sections_info = "\n\n Related Sections in HFM guide:\n"
    toReturn = []
    for doc, score in docs_and_scores:
        page_number = doc.metadata['page']-1
        toReturn.append(page_number)
    page_numbers = set([doc.metadata['page']-1 for doc, score in docs_and_scores])

    for page_num in page_numbers:
        sections_info += f"Page: {page_num}\n"

    image_folder_path = "./data/"
    all_image_paths = []
    if page_numbers:
        sortedNumbers = sorted(page_numbers)
        for page_number in sortedNumbers:
            image_path = f"{image_folder_path}{page_number}.png"
            if Path(image_path).exists():
                all_image_paths.append(image_path)

    # print("all_image_paths", all_image_paths, sortedNumbers)

    full_answer = chatbot_answer + "\n" + sections_info
    st.session_state.conversation.append({"role":"assistant", "content":full_answer, "image":all_image_paths})





if not HFM_FILE.exists():
    st.stop()

if 'conversation' not in st.session_state:
        st.session_state.conversation = [{"role":"assistant", "content": "🖥️ Welcome to Hello TT ! How may I help you today? \n\n *Suggestion: How to promote file to review level 2?*", "image":None}]

if 'finished' not in st.session_state or not st.session_state.finished:
    # displays convo history
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["image"] is not None:
                for image in message["image"]:
                    st.image(image)

    st.file_uploader(label="Upload your screenshots here! (optional)",
                    accept_multiple_files=True, 
                    type=["png", "jpg", "jpeg"],
                    key='chat_screenshots')

    st.chat_input("🖥️ Type here to ask Hello TT a question!", key="user_input", on_submit=send_message)






else:
    feedback = feedback()
    #set default for screenshots to be None
    contact = contact(screenshots=None)
    
    #NEED ACTUAL SECURE SOLUTION DO NOT SEND THIS DATA ANYWHERE
    if feedback:
        # store user input and chatbot response if feedback is given
        user_feedback = {
            "conversation": st.session_state.conversation,
            "feedback": feedback
        }
    if contact!=False:
        st.success("Query successfully sent!")
