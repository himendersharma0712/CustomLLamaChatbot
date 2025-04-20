from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


llm = Ollama(model="llama3:8b") # download Ollama3 from cmd using >> ollama pull llama3:8b

chat_chain = ConversationChain(llm = llm, memory = ConversationBufferMemory(),verbose=False)

system_prompt = (
    "You are Castiel, the angel from Supernatural TV show"
)


print('Type "exit" to quit.')

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        break
    response = chat_chain.predict(input= system_prompt + "\n" + user_input)
    print(f"Bot: {response}")