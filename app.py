#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ChatWithYourData Bot
--------------------
A retrieval-augmented chatbot built using LangChain, OpenAI, and Panel.
This script allows users to chat with their own documents using RAG (Retrieval-Augmented Generation).
"""

import os
import sys
import datetime
import panel as pn
import param
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path
sys.path.append('../..')

# OpenAI Key
import openai
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Select model version
current_date = datetime.datetime.now().date()
llm_name = "gpt-3.5-turbo-0301" if current_date < datetime.date(2023, 9, 2) else "gpt-3.5-turbo"

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain

"""
----------------------------
 Utility Functions
----------------------------
"""

def load_db(file_path: str, chain_type: str, k: int):
    """
    Load a PDF file, create embeddings, and store in ChromaDB for retrieval.
    """
    # Load documents
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # Create embeddings and persist to local Chroma directory
    persist_directory = "docs/chroma"
    embeddings = OpenAIEmbeddings()

    # Create or load existing Chroma vector store
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # Define retriever
    retriever = db.as_retriever(search_kwargs={"k": k})

    # Create chatbot chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )

    return qa

"""
----------------------------
Chatbot Class
----------------------------
"""

class ChatBot(param.Parameterized):
    """Interactive chatbot connected to user documents."""

    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])

    def __init__(self, **params):
        super().__init__(**params)
        self.panels = []
        self.loaded_file = "docs/cs229_lectures/MachineLearning-Lecture01.pdf"
        self.qa = load_db(self.loaded_file, "stuff", 4)

    def call_load_db(self, count):
        """Load or reload the vector database from file input."""
        if count == 0 or file_input.value is None:
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

        file_input.save("temp.pdf")
        self.loaded_file = file_input.filename
        button_load.button_style = "outline"
        self.qa = load_db("temp.pdf", "stuff", 4)
        button_load.button_style = "solid"
        self.clear_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        """Run the chatbot conversation."""
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)

        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result.get("generated_question", "")
        self.db_response = result.get("source_documents", [])
        self.answer = result['answer']

        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600,
                                               style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''
        return pn.WidgetBox(*self.panels, scroll=True)

    def clear_history(self, _=0):
        """Clear the chat history."""
        self.chat_history = []

    @param.depends('db_query')
    def get_last_query(self):
        """Return last database query info."""
        if not self.db_query:
            return pn.pane.Markdown("No DB accesses yet.")
        return pn.pane.Markdown(f"**DB Query:** {self.db_query}")

    @param.depends('db_response')
    def get_sources(self):
        """Return retrieved source documents."""
        if not self.db_response:
            return pn.pane.Markdown("No sources retrieved yet.")
        return pn.WidgetBox(*[pn.pane.Str(str(doc)) for doc in self.db_response], width=600, scroll=True)


""" 
----------------------------
Panel App Setup
----------------------------
"""

pn.extension()
cb = ChatBot()

file_input = pn.widgets.FileInput(accept='.pdf')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(cb.clear_history)
inp = pn.widgets.TextInput(placeholder='Enter text here…')

bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.convchain, inp)
img_pane = pn.pane.Image('./img/convchain.jpg', width=400)

tab1 = pn.Column(pn.Row(inp), pn.layout.Divider(),
                 pn.panel(conversation, loading_indicator=True, height=300))
tab2 = pn.Column(pn.panel(cb.get_last_query), pn.layout.Divider(), pn.panel(cb.get_sources))
tab3 = pn.Column(pn.pane.Markdown("Chat History"), pn.layout.Divider())
tab4 = pn.Column(pn.Row(file_input, button_load, bound_button_load),
                 pn.Row(button_clearhistory, pn.pane.Markdown("Clears chat history.")),
                 pn.layout.Divider(), pn.Row(img_pane))

dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
    pn.Tabs(('Conversation', tab1), ('Database', tab2), ('History', tab3), ('Configure', tab4))
)

# To run: dashboard.show()
