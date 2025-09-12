import numpy as np
import os
import glob
from dotenv import load_dotenv
import logging
# langchain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from chromadb.api.models.Collection import Collection

#-------------------------------------------------------------------------
class LoadDocuments :
    #-------------------------------------------------------------------------
    @staticmethod
    def create_documents_list(md_filespath : str) -> list[Document]:

        full_md_filespath : str = f'{md_filespath}/*.md'

        text_loader_kwargs : dict[str,str] = {'encoding': 'utf-8'}
        collection_documents_list : list[Document] = []


        direct_md_files = glob.glob(full_md_filespath)
        #----------------------------------------
        for md_file in direct_md_files:
            loader = TextLoader(file_path=md_file, **text_loader_kwargs)
            docs : list[Document] = loader.load()
            for doc in docs:
                doc.metadata["doc_type"] = "root"
                collection_documents_list.append(doc)
        #----------------------------------------

        # in case we ever decide to place files in subfolders ...
        folders_list : list[str] = [
            file 
            for file in glob.glob("md/*") 
            if os.path.isdir(file)
        ]

        #----------------------------------------
        for curr_folder_path in folders_list:
            curr_folder_name : str = os.path.basename(curr_folder_path)
            curr_directory_loader = DirectoryLoader(
                path            = curr_folder_path, 
                glob            = "**/*.md", 
                loader_cls      = TextLoader, 
                loader_kwargs   = text_loader_kwargs
            )
            curr_folder_docs : list[Document] = curr_directory_loader.load()
            #----------------------------------------
            for loop_doc in curr_folder_docs:
                loop_doc.metadata["doc_type"] = curr_folder_name
                collection_documents_list.append(loop_doc)
            #----------------------------------------
        #----------------------------------------

        return collection_documents_list
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    @staticmethod
    def create_chunk_list(collection_documents_list : list[Document]) -> list[Document]:
        text_splitter = CharacterTextSplitter(
            chunk_size      = 1000, 
            chunk_overlap   = 200
        )

        chunks_list : list[Document] = text_splitter.split_documents(
            documents = collection_documents_list
        )

        return chunks_list
    #-------------------------------------------------------------------------      
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
class VecDBCreation :
    #-------------------------------------------------------------------------
    def __init__(self, db_name : str = "vector_db", openai_api_key : str|None = None ) -> None :
        load_dotenv()

        self.openai_api_key : str = openai_api_key if openai_api_key else os.getenv('OPENAI_API_KEY')
        self.db_name = db_name
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def create_vector_store(self, chunks_list : list[Document]) -> Chroma :

        # better embeddings - this is more accurate
        # openai_api_key : str = os.getenv('OPENAI_API_KEY')
        embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)


        # delete if already exists
        if os.path.exists(self.db_name):
            Chroma(
                persist_directory  = self.db_name, 
                embedding_function = embeddings
            ).delete_collection()


        # create the vector DB
        chroma_vectorstore : Chroma = Chroma.from_documents(
            documents           = chunks_list, 
            embedding           = embeddings, 
            persist_directory   = self.db_name
        )
        
        logging.info(f"Vectorstore created with {chroma_vectorstore._collection.count()} documents")

        return chroma_vectorstore
    #------------------------------------------------------------------------- 
    #-------------------------------------------------------------------------
    def check_collection(self, vectorstore : Chroma) -> Collection:
        # let's check if we can fetch one vector and then check its dimensions
        
        collection : Collection = vectorstore._collection

        sample_one_embedding : np.ndarray = collection.get(
            limit   = 1, 
            include = ["embeddings"]
        )["embeddings"][0]

        dimensions_of_one_embedding : int = len(sample_one_embedding)

        # print(f"The vectors have {dimensions_of_one_embedding:,} dimensions")
        logging.info(f"The vectors have {dimensions_of_one_embedding:,} dimensions")
        
        return collection # returning this just in case we want to inspect further
    #-------------------------------------------------------------------------       
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def main():

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s - %(levelname)s - %(message)s"
    )

    knowledge_base : str = "./md"

    result_collection_documents_list : list[Document] = LoadDocuments.create_documents_list(
        md_filespath = knowledge_base
    )

    result_chunks_list : list[Document] = LoadDocuments.create_chunk_list(
        collection_documents_list = result_collection_documents_list
    )

    #---------

    vecdb_actions = VecDBCreation()
    result_vectorstore : Chroma = vecdb_actions.create_vector_store(chunks_list = result_chunks_list)
    vecdb_actions.check_collection(vectorstore=result_vectorstore)
#-------------------------------------------------------------------------

if __name__ == "__main__":
    main()