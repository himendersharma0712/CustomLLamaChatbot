from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import ConversationChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from prompt_toolkit import prompt
from prompt_toolkit.patch_stdout import patch_stdout
import threading
import warnings
import pickle

warnings.filterwarnings('ignore')

llm = OllamaLLM(model="llama2", temperature=0.3)

# Set up a persistent memory for important memories
important_memories = []
important_memory_file = "max_important.pkl"

# Try to load important memories if they exist
try:
    with open(important_memory_file, "rb") as f:
        important_memories = pickle.load(f)
except FileNotFoundError:
    important_memories = []

# Load PDF and setup retrieval
pdf_paths = ["D:/CustomLLamaBot/book2.pdf"]
all_docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs = loader.load()
    all_docs.extend(docs)

# Split documents into chunks for retrieval and summary
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(all_docs)

# Create embeddings and vector store for Q/A
embeddings = OllamaEmbeddings(model="llama2")
db = Chroma.from_documents(split_docs, embeddings)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Generate essence from first few chunks
intro_text = "\n".join([doc.page_content for doc in split_docs[:3]])
essence_prompt = (
    "Summarize the emotional themes from the following text ''for a change in personality'':\n\n" + intro_text
)

def fetch_book_answer(query):
    response = qa_chain.invoke(query)
    print(f"\nMax (from book): {response['result']}\n")

book_context = ""
for chunk in llm.stream(essence_prompt):
    book_context += chunk

# Define the template with Max's therapist and memory style
template = """
You are Max, a kind and understanding AI therapist. You remember important things the user tells you and provide emotional support and thoughtful insights. Your goal is to help the user feel heard and understood.

You always remember these things about the user:
"{important_stuff}"

Current conversation:
{history}
User: {input}
Max:"""

prompt_template = PromptTemplate(
    input_variables=["important_stuff", "history", "input", "book_context"],
    template=template
)

# Handle user interaction
print("Type 'exit' to quit. Use prefix 'book:' to query PDF content explicitly.\n")

history = ""  # Initialize conversation history

with patch_stdout():
    while True:
        user_input = prompt("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Max: Take care. I'll always be here when you need someone to talk to. 🌱")
            break

        # Check if the user wants Max to remember something
        if user_input.lower().startswith("remember this"):
            memory_content = user_input[len("remember this"):].strip()
            if memory_content:
                important_memories.append(memory_content)
                with open(important_memory_file, "wb") as f:
                    pickle.dump(important_memories, f)
                print(f"Max: I'll remember that... \"{memory_content}\" is important to me, and to you. 💖")
            else:
                print("Max: What would you like me to remember? Please tell me after 'remember this'. 📝")
            continue

        if user_input.lower().startswith("book:"):
            query = user_input[len("book:"):].strip()
            print("Max: Let me check that book for you...\n")
            thread = threading.Thread(target=fetch_book_answer, args=(query,))
            thread.start()
            continue

        # Collect history as the temporary memory for the conversation
        messages = history.split("\n")
        history = "\n".join(messages[-5:])  # Keep the last 5 lines for the current conversation

        # Format the prompt with memories, history, and user input
        formatted_prompt = prompt_template.format(
            important_stuff="\n".join(important_memories),
            book_context=book_context,
            history=history.strip(),
            input=user_input
        )

        print("Max: ", end="", flush=True)
        full_response = ""
        for chunk in llm.stream(formatted_prompt):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()

        # Update conversation history
        history += f"User: {user_input}\nMax: {full_response}\n"
