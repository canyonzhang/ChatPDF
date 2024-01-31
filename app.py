import streamlit as st
import sys
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS # Using the FAISS vector store, langchain supports many different types of vector stores
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# automatically reads variables from the .env file (openai api key)
load_dotenv()

def main():
    st.header("Chat with PDF 💬")

    # streamlit pdf file uploader
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    # st.write(pdf)

    # Check if there is a pdf uploaded
    if pdf is not None:
        # use PdfReader from PyPDF2 to read the pdf file
        pdf_reader = PdfReader(pdf)
        text = ""
        # for each page in the pdf_reader, accumulate the text
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)

         # Create a  recursive character text splitter from langchain.text_splitter to split pdf into 1k token chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        
        # split the pdf and return it into a chunks array
        chunks = text_splitter.split_text(text=text)
 
        # Extract the name of the file to write it into our vector database
        file_name = pdf.name[:-4] # get rid of the .pdf extension
        st.write(f'{file_name}')

        # Check if the file exists on our disk, if it does, load it and load its vector representation from our vectorstore 
        if os.path.exists(f"{file_name}.pkl"):
            with open(f"{file_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f) # read the file that exists on disk and store it in a VectorStore variable
            st.write("Embeddings loaded from the disk")
        # # If the file doesn't exist in our disk, embed it and store it into our vector database
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{file_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        
        # Await the user query 
        query = st.text_input("Ask questions about your PDF file:")

        # perform a similarity search using the users embedded query
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            # initialize an instance of OpenAI's LLM wrapper (langchain)
            llm = OpenAI(model_name="gpt-3.5-turbo-0301")
            # create a q and a chain (langchain)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            # get the cost for the query we run
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            # write the generated response returned from the OpenAI LLM instance in combination with the context (pdf) we provided
            st.write(response)

if __name__ == '__main__':
    main()