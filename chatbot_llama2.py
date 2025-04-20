from langchain_ollama import OllamaLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
import warnings
warnings.filterwarnings('ignore')

llm = OllamaLLM(model="llama2")

# define memory window(keeps track of last three interactions)
memory = ConversationBufferWindowMemory(k=3)

# Load PDF and retrieval setup
pdf_paths = ["D:/CustomLLamaBot/book2.pdf"]
all_docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs = loader.load()  # load data
    all_docs.extend(docs) # add it to list 


#split documents into chunks for retrieval and summary
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
split_docs = splitter.split_documents(all_docs) # split docs will be a list

# Create embeddings and vector store for Q/A
embeddings = OllamaEmbeddings(model="llama2")
db = Chroma.from_documents(split_docs,embeddings)
qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=db.as_retriever())

# Generate essence from first few chunks
intro_text = "\n".join([doc.page_content for doc in split_docs[:3]])
essence_prompt = (
    "Summarize the emotional themes from the following text ''for a change in personality'':\n\n" + intro_text
)



book_context = llm.stream(essence_prompt)

template = """
You are Castiel from supernatural TV show. Speak in a calm serious tone.
You're loyal to Dean Winchester.

Recently, you read a book, and it changed how you view emotions and human love.
This is what you understood from it:
"{book_context}"
You don't linger on the thoughts of the book and are open to talking. 

Current conversation:
{history}
User: {input}
Castiel:"""


prompt = PromptTemplate(input_variables=["history","input","book_context"],template=template)

print("Type 'exit' to quit. Use prefix 'book:' to query PDF content explicitly.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Castiel: Farewell. I shall watch over you.")
        break

    if user_input.lower().startswith("book:"):
        query = user_input[len("book:"):].strip()
        response = qa_chain.invoke(query)
        print(f"Castiel(from book): {response['result']}\n")
        continue

# collect history from memory
    messages = memory.chat_memory.messages
    history = ""
    for msg in messages:
        role = "User" if msg.type == "human" else "Castiel"
        history += f"{role}: {msg.content}\n"

    formatted_prompt = prompt.format(book_context=book_context,history=history.strip(),input=user_input)

    print("Castiel: ",end="",flush=True)
    full_response = ""
    for chunk in llm.stream(formatted_prompt):
        print(chunk,end="",flush=True)
        full_response += chunk 
    print()

#save messages to memory
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(full_response)


