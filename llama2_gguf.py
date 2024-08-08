
from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain import PromptTemplate
import gradio as gr
import time

my_prompt = """
You are a respectful and helpful AI Chat assistant that responds to user questions
Take context from chat history to give better answers; Chat History: {chat_history}.
The following is the user's question, answer it accurately, User Question: {user_input}
After Answering user's question stop.
"""

def SetPrompt():
    prompt = PromptTemplate(template = my_prompt, input_variables = ['user_input',"chat_history"])
    return prompt

def LoadModel():
    llm = CTransformers(
        model ='unsloth.Q4_K_M.gguf',
        model_type = 'llama',
        max_new_tokens = 256,
        temperature = 0.2,
        gpu_layers = 0
    )

    return llm


def ChainPipeline(mem):
    llm = LoadModel()
    qa_prompt = SetPrompt()
    qa_chain = LLMChain(
        prompt = qa_prompt,
        llm = llm,
        memory = mem
    )

    return qa_chain



memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5,
    return_messages=True
)


llmChain = ChainPipeline(memory)

# user_input = "Hello, I am Nameer."

# llmResponse = llmChain.run({"user_input": user_input})


def bot(user_input):
    llmResponse = llmChain.run({"user_input": user_input})
    return llmResponse

with gr.Blocks(title = "Chatbot") as demo:
    gr.Markdown("Chatbot")
    chatbot = gr.Chatbot([],elem_id = "chatbot", height = 700)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def GetResponse(message, chatHistory):
        getResp = bot(message)
        chatHistory.append((message, getResp))
        time.sleep(2)
        return "", chatHistory
    
    msg.submit(GetResponse, [msg,chatbot], [msg,chatbot])

demo.launch()
