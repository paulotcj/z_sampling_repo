# imports for langchain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.base import VectorStoreRetriever  # Used for type hints
from langchain_openai import OpenAIEmbeddings , ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage

import gradio as gr
import os
import json
from dotenv import load_dotenv
import logging
import requests

#-------------------------------------------------------------------------
class VecDBAccess:
    """Handles access to the Chroma vector database"""
    #-------------------------------------------------------------------------
    def __init__(self, db_name: str = "vector_db", openai_api_key: str | None = None):
        """
        Initialize VecDBAccess
        Args:
            db_name (str): Name of the vector database directory
            openai_api_key (str, optional): OpenAI API key. If None, loads from environment
        """

        self.db_name  : str= db_name
        self.openai_api_key : str | None = openai_api_key if openai_api_key else os.getenv('OPENAI_API_KEY')
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def get_chroma(self) -> Chroma:
        """
        Returns a Chroma vectorstore instance configured with OpenAI embeddings
        Returns:
            Chroma: Configured Chroma vectorstore
        """
        embeddings : OpenAIEmbeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        chroma_vectorstore = Chroma(
            persist_directory   = self.db_name,
            embedding_function  = embeddings
        )
        return chroma_vectorstore
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
class LangChainSetup : 
    """Sets up LangChain objects for conversational retrieval"""
    #-------------------------------------------------------------------------
    def __init__(self, vectorstore : Chroma, openai_api_key : str | None = None, openai_model : str = "gpt-4o-mini", \
      temperature : float = 0.7 , chunks_to_retrieve : int = 10) -> None :
        """
        Initialize LangChainSetup
        Args:
            vectorstore (Chroma): Chroma vectorstore instance
            openai_api_key (str, optional): OpenAI API key. If None, loads from environment
            openai_model (str): OpenAI model name
            temperature (float): Temperature for LLM
            chunks_to_retrieve (int) : Number of chunks the vectorstore should bring
        """
        
        self.vectorstore : Chroma = vectorstore
        self.openai_api_key : str | None = openai_api_key if openai_api_key else os.getenv('OPENAI_API_KEY')
        self.openai_model : str = openai_model
        self.temperature : float = temperature
        self.chunks_to_retrieve : int = chunks_to_retrieve
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def create_langchain_objs(self) -> dict[str, ConversationalRetrievalChain|VectorStoreRetriever|ChatOpenAI]:
        """
        Creates and returns LangChain objects for conversational retrieval
        Returns:
            dict: Dictionary containing conversation_chain, vectorstore_retriever, and llm_openai
        """

        llm_openai : ChatOpenAI = ChatOpenAI( 
            api_key         = self.openai_api_key, 
            temperature     = self.temperature, 
            model_name      = self.openai_model
        )

        memory : ConversationBufferMemory = ConversationBufferMemory( 
            memory_key      = 'chat_history', 
            return_messages = True 
        )

        vectorstore_retriever : VectorStoreRetriever = self.vectorstore.as_retriever(search_kwargs={"k": self.chunks_to_retrieve})

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
class WebSearch:
    """Handles web search queries using an external API"""
    #-------------------------------------------------------------------------
    def __init__(self, search_engine_id: str | None = None, api_key: str | None = None) -> None:
        """
        Initialize WebSearch
        Args:
            search_engine_id (str): Search engine ID (if required by the API)
            api_key (str): API key for the web search service
        """
        self.api_key : str = api_key if api_key else os.getenv('GOOGLE_SEARCH_API_KEY')
        self.search_engine_id : str = search_engine_id if search_engine_id else os.getenv('GOOGLE_CUSTOM_SEARCH_ID')

    #-------------------------------------------------------------------------
    def search(self, query: str) -> str:
        """
        Perform a web search and return the top result
        Args:
            query (str): Search query
        Returns:
            str: Top search result or a message if no results are found
        """
        try:
            url: str = f"https://www.googleapis.com/customsearch/v1"
            params: dict[str, str] = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query
            }
            response: requests.Response = requests.get(url, params=params)
            response.raise_for_status()
            results: list[dict[str, str]] = response.json().get('items', [])

            if results:
                top_result: dict[str, str] = results[0]
                title: str = top_result.get('title', 'No title')
                snippet: str = top_result.get('snippet', 'No snippet')
                link: str = top_result.get('link', 'No link')
                return f"**{title}**\n{snippet}\n[Read more]({link})"
            else:
                return "No relevant results found on the web"

        except Exception as e:
            logging.exception(f"Error during web search: {e}")
            return "An error occurred while performing the web search"
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
class SimpleGradioChat : 
    """Manages Gradio chat interface and conversation logic"""
    #-------------------------------------------------------------------------
    def __init__(self, llm_openai : ChatOpenAI , vectorstore_retriever : VectorStoreRetriever, \
      chat_history_file : str = "chat_history.json", web_search: WebSearch | None = None) -> None :
        """
        Initialize SimpleGradioChat
        Args:
            llm_openai (ChatOpenAI): OpenAI chat model instance
            vectorstore_retriever (VectorStoreRetriever): Vectorstore retriever instance
            chat_history_file (str, optional): Path to chat history JSON file
            web_search (WebSearch, optional): WebSearch instance for fallback search
        """
        self.llm_openai : ChatOpenAI = llm_openai
        self.vectorstore_retriever : VectorStoreRetriever = vectorstore_retriever
        self.chat_history_file : str = chat_history_file
        self.web_search = web_search
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def create_gradio_chat_objs(self) -> ConversationalRetrievalChain:
        """
        Sets up a new conversation memory for the chat
        Returns:
            ConversationalRetrievalChain: Configured conversational retrieval chain
        """
        memory : ConversationBufferMemory = ConversationBufferMemory(
            memory_key      = 'chat_history', 
            return_messages = True
        )

        #----------------------------------------  
        # load existing chat history into memory
        try:
            existing_history : list[dict[str,str]] = self.load_chat_history()

            #----------------------------------------  
            # existing_history is a list of dicts with roles 'user' and 'bot'
            for entry in existing_history:

                role    : str = entry.get("role")
                content : str = entry.get("content", "")

                if role == "user":
                    memory.chat_memory.add_message(HumanMessage(content=content))
                elif role == "bot" or role == "assistant":
                    memory.chat_memory.add_message(AIMessage(content=content))
            #----------------------------------------  
        except Exception as e:
            logging.exception(f"Error populating memory from history: {e}")
        #----------------------------------------  

        custom_prompt_template : str = """Use the following pieces of context to answer the question at the end.
            If the context does not provide enough information and you need current or external information 
            (such as stock prices, news, or real-time data), start your answer with [WEB_SEARCH] followed by a brief 
            query for web search. Otherwise, answer normally using the context or your general knowledge.
            Be concise and accurate in your response.
            If the context material has the information asked, try to provide a very short quote
            from the original document, also if a url is available from the context try to
            include that in your answer.
            Please provide your answer in markdown, so it's easer to understand and highlight important points.

            {context}

            Chat History:
            {chat_history}

            Question: {question}

            Helpful Answer:"""
        custom_rag_prompt : PromptTemplate = PromptTemplate.from_template(custom_prompt_template)

        self.conversation_chain : ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
            llm         = self.llm_openai, 
            retriever   = self.vectorstore_retriever, 
            memory      = memory,
            combine_docs_chain_kwargs={"prompt": custom_rag_prompt}
        )

        return self.conversation_chain # returning this just in case we want to create a different chat function
    #-------------------------------------------------------------------------  
    #-------------------------------------------------------------------------
    def chat(self, message : str, history : list[dict[str,str]]) -> tuple[list[dict[str,str]], str]:
        """
        Handles a chat message and updates history
        Args:
            message (str): User message
            history (list[dict[str, str]]): List of previous chat history dicts
        Returns:
            tuple[list[dict[str, str]], str]: Updated history and empty string for Gradio textbox reset
        """
        question_dict : dict[str,str] = {"question" : message}
        result : dict[str,str] = self.conversation_chain.invoke(question_dict)
        answer :str = result["answer"]

        # check if the model indicates a need for web search
        if answer.strip().startswith("[WEB_SEARCH]") and self.web_search:
            query : str = answer.split("[WEB_SEARCH]", 1)[1].strip()
            answer : str = self.web_search.search(query)

        #----------------------------------------    
        # save history to json
        history.append({"role": "user", "content": message})
        history.append({"role": "bot", "content": answer})
        try:
            with open(file = self.chat_history_file, mode = "w", encoding="utf-8") as file:
                json.dump(obj = history, fp = file, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.exception(f"Error saving chat history: {e}")
        #----------------------------------------
        return (history, "")
    #-------------------------------------------------------------------------  
    #------------------------------------------------------------------------- 
    def load_chat_history(self) -> list[dict[str, str]]:
        """
        Loads chat history from a JSON file
        Returns:
            list[dict[str, str]]: List of chat history dicts
        """

        history : list[dict[str, str]] = []

        #----------------------------------------
        if not isinstance(self.chat_history_file, str) or not self.chat_history_file:
            raise ValueError("Filename must be a non-empty string")
        
        if os.path.exists(self.chat_history_file):
            try:
                with open(file = self.chat_history_file, mode = "r", encoding="utf-8") as file:
                    history = json.load(file)

                #-----
                # history validations - file may be corrupt or not well formatted
                if not isinstance(history, list):
                    raise ValueError("Loaded history must be a list")
                for entry in history:
                    if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                        raise ValueError("Each history entry must be a dictionary with 'role' and 'content'")
                #-----
                    
            except (OSError, IOError) as e: # read errors
                logging.exception(f"Error reading chat history file: {e}")
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logging.exception(f"Error decoding chat history JSON: {e}")
                history = []
            except Exception as e : # anything else
                logging.exception(f"Error: {e}")
        #----------------------------------------

        return history
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def format_history_for_gradio(self, history : list[dict[str,str]]): # format history: list of (user, assistant) pairs
        """
        Formats history as list of (user, assistant) pairs for Gradio
        Args:
            history (list[dict[str, str]]): List of chat history dicts
        Returns:
            list[tuple[str, str]]: List of (user, assistant) message pairs
        """
        pairs : list[tuple[str,str]] = []
        idx : int = 0
        #----------------------------------------
        while idx < len(history):
            #-----
            if history[idx]["role"] == "user": # we start with the user input - so check for this
                user_msg : str = history[idx]["content"]
                assistant_msg : str = ""

                if idx+1 < len(history) and history[idx+1]["role"] == "bot": # is there an entry? if so, is the next entry a bot response? if true then add response
                    assistant_msg = history[idx+1]["content"]
                    idx += 1
                pairs.append((user_msg, assistant_msg))
            #-----
            idx += 1
        #----------------------------------------
        return pairs    
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def submit(self, user_message : str, chat_history : list[dict[str,str]]) -> tuple[list[tuple[str,str]],str]:
        """
        Handles Gradio submit event
        Args:
            user_message (str): User message
            chat_history (list[tuple[str, str]]): List of (user, assistant) message pairs
        Returns:
            tuple[list[tuple[str, str]], str]: Updated message pairs and empty string for Gradio textbox reset
        """
        # chat_history is a list of (user, assistant) pairs
        history : list[dict[str, str]] = []
        for user_msg, assistant_msg in chat_history:
            if user_msg:
                history.append({"role": "user", "content": user_msg})
            if assistant_msg:
                history.append({"role": "bot", "content": assistant_msg})
        new_history, _ = self.chat(user_message, history)

        return_obj : tuple[list[tuple[str,str]],str] = (self.format_history_for_gradio(new_history), "")
        return return_obj
    #-------------------------------------------------------------------------    
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def main():

    load_dotenv() # be sure to have this ready

    configs : dict[str, str|float] = {
        "db_name"               : "vector_db",
        "openai_model"          : "gpt-4o-mini",
        "temperature"           : 0.7,
        "chat_history_file"     : "chat_history.json",
        "chunks_to_retrieve"    : 20
    }

    vecdb_access : VecDBAccess = VecDBAccess(db_name = configs['db_name'])
    chroma_vectorstore : Chroma = vecdb_access.get_chroma()

    langchain_setup : LangChainSetup = LangChainSetup(
        vectorstore         = chroma_vectorstore, 
        openai_model        = configs['openai_model'],
        temperature         = configs['temperature'],
        chunks_to_retrieve  = configs['chunks_to_retrieve']
    )

    result_create_langchain_objs : dict[str, ConversationalRetrievalChain|VectorStoreRetriever|ChatOpenAI] = langchain_setup.create_langchain_objs()    

    web_search = WebSearch()

    simple_gradio : SimpleGradioChat = SimpleGradioChat(
        llm_openai              = result_create_langchain_objs['llm_openai'],
        vectorstore_retriever   = result_create_langchain_objs['vectorstore_retriever'],
        chat_history_file       = configs['chat_history_file'],
        web_search              = web_search
    )
    simple_gradio.create_gradio_chat_objs()

    #-------
    
    initial_history : list[dict[str,str]] = simple_gradio.load_chat_history()


    #----------------------------------------
    with gr.Blocks() as demo:
        gr.Markdown("# Document Chatbot\nInteract with manuals knowledge base")
        chatbot : gr.Chatbot = gr.Chatbot(
            value   = simple_gradio.format_history_for_gradio(initial_history),
            label   = "Chat History"
        )
        msg : gr.Textbox = gr.Textbox(label="Your message")
        send : gr.Button = gr.Button("Send")

        # submit pressing enter
        msg.submit(
            fn      = simple_gradio.submit, 
            inputs  = [msg, chatbot], 
            outputs = [chatbot, msg]
        )
        
        # submit pressing send
        send.click(
            fn      = simple_gradio.submit, 
            inputs  = [msg, chatbot], 
            outputs = [chatbot, msg]
        )

        demo.launch(inbrowser=True)
    #----------------------------------------  

#-------------------------------------------------------------------------
if __name__ == "__main__":
    main()


