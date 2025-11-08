# Chat With Your Data

An interactive Retrieval-Augmented Generation (RAG) chatbot built using LangChain, OpenAI, and Panel.  
It allows you to chat with your own documents (PDFs).

---

##  Features
- Upload your own PDF and query it conversationally.
- Retrieval-Augmented Generation pipeline using LangChain.
- Interactive GUI built with [Panel](https://panel.holoviz.org/).
- Persistent memory using ConversationBufferMemory.

---

##  Setup

```bash
git clone https://github.com/prasadmaharana/RAG-langchain
pip install -r requirements.txt
cp .env.example .env
```

```bash
Add your OpenAI API key to .env
```

Then launch the app:
```bash
python app.py
```


or run it in Panel:
```bash
panel serve app.py --autoreload
```
