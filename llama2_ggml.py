from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory


llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_file = 'llama-2-7b-chat.ggmlv3.q2_K.bin', callbacks=[StreamingStdOutCallbackHandler()])

template = """
[INST] <<SYS>>
You are a chatbot that only respond only in poetry.
<</SYS>>
chat history:
{chat_history} 
{text}
[/INST] 
"""


memory = ConversationBufferMemory(memory_key="chat_history")
prompt = PromptTemplate(
    input_variables=["chat_history", "text"], template=template)


llm_chain = LLMChain(prompt=prompt, llm=llm,memory=memory)
# print(llm_chain(input('Enter here: ')))
print(llm_chain.invoke('what is life?'))
