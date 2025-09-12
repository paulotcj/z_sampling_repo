# imports for langchain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings , ChatOpenAI
from langchain_chroma import Chroma

import gradio as gr
import os
from dotenv import load_dotenv

#-------------------------------------------------------------------------
class VecDBAccess:
    #-------------------------------------------------------------------------
    def __init__(self, db_name: str = "vector_db", openai_api_key: str | None = None):
        load_dotenv()
        self.db_name  : str= db_name
        self.openai_api_key : str = openai_api_key if openai_api_key else os.getenv('OPENAI_API_KEY')
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def get_chroma(self) -> Chroma:
        embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        chroma_vectorstore = Chroma(
            persist_directory   = self.db_name,
            embedding_function  = embeddings
        )
        return chroma_vectorstore
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
class LangChainSetup : 
    #-------------------------------------------------------------------------
    def __init__(self, vectorstore : Chroma, openai_api_key : str = None, openai_model : str = "gpt-4o-mini", \
      temperature : float = 0.7) -> None :
        load_dotenv()
        self.vectorstore : Chroma = vectorstore
        self.openai_api_key : str = openai_api_key if openai_api_key else os.getenv('OPENAI_API_KEY')
        self.openai_model : str = openai_model
        self.temperature : float = temperature
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def create_langchain_objs(self) -> dict[str, ConversationalRetrievalChain|VectorStoreRetriever|ChatOpenAI]:

        llm_openai : ChatOpenAI = ChatOpenAI( 
            api_key         = self.openai_api_key, 
            temperature     = self.temperature, 
            model_name      = self.openai_model
        )

        memory : ConversationBufferMemory = ConversationBufferMemory( 
            memory_key      = 'chat_history', 
            return_messages = True 
        )

        vectorstore_retriever : VectorStoreRetriever = self.vectorstore.as_retriever(search_kwargs={"k": 25})

        conversation_chain : ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
            llm         = llm_openai, 
            retriever   = vectorstore_retriever, 
            memory      = memory
        )

        return_dict : dict[str, ConversationalRetrievalChain|VectorStoreRetriever|ChatOpenAI] = {
            'conversation_chain'    : conversation_chain,
            'vectorstore_retriever' : vectorstore_retriever,
            'llm_openai'            : llm_openai
        }

        return return_dict
    #-------------------------------------------------------------------------    
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
class SimpleGradioChat : 
    #-------------------------------------------------------------------------
    def __init__(self, llm_openai : ChatOpenAI , vectorstore_retriever : VectorStoreRetriever) -> None :
        self.llm_openai : ChatOpenAI = llm_openai
        self.vectorstore_retriever : VectorStoreRetriever = vectorstore_retriever
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def create_gradio_chat_objs(self) -> ConversationalRetrievalChain:
        # set up a new conversation memory for the chat
        memory = ConversationBufferMemory(
            memory_key      = 'chat_history', 
            return_messages = True
        )

        self.conversation_chain : ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
            llm         = self.llm_openai, 
            retriever   = self.vectorstore_retriever, 
            memory      = memory
        )

        return self.conversation_chain # returning this just in case we want to create a different chat function
    #-------------------------------------------------------------------------  
    #-------------------------------------------------------------------------
    def chat(self, message : str, history):
        question_dict : dict[str,str] = {"question" : message}
        result = self.conversation_chain.invoke(question_dict)
        
        answer = result["answer"]

        #----------------------------------------    
        # Save history to a file
        with open("chat_history.txt", "w") as f:
            #----------------------------------------    
            for msg in history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                f.write(f"{role.capitalize()}: {content}\n")
            #----------------------------------------    

            f.write(f"User: {message}\nBot: {answer}\n")
        #----------------------------------------    

        return answer    
    #-------------------------------------------------------------------------  
    #------------------------------------------------------------------------- 
    def load_chat_history(self, filename="chat_history.txt") -> list[dict[str, str]]:
        history : list[dict[str, str]] = []
        if os.path.exists(filename):
            with open(filename, "r") as f:
                for line in f:
                    if ": " in line:
                        role, content = line.strip().split(": ", 1)
                        history.append({"role": role.lower(), "content": content})
        return history  
    #-------------------------------------------------------------------------       
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def main():

    vecdb_access = VecDBAccess()
    chroma_vectorstore : Chroma = vecdb_access.get_chroma()

    langchain_setup = LangChainSetup(vectorstore=chroma_vectorstore)
    result_create_langchain_objs  = langchain_setup.create_langchain_objs()    

    simple_gradio = SimpleGradioChat(
        llm_openai              = result_create_langchain_objs['llm_openai'],
        vectorstore_retriever   = result_create_langchain_objs['vectorstore_retriever']
    )
    conversation_chain : ConversationalRetrievalChain = simple_gradio.create_gradio_chat_objs()



    view = gr.ChatInterface(
        fn      = simple_gradio.chat, 
        type    = "messages",
    ).launch(inbrowser=True)   
#-------------------------------------------------------------------------
if __name__ == "__main__":
    main()


