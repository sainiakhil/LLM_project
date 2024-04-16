# LLM_project
# Retrieval Augmented Generation (RAG) with Open Source LLM and LangChain

# Title: 
Integration of Retrieval Augmented Generation (RAG) with Open Source LLM and LangChain for Autism Intervention Research.
# Objective: 
This assignment aims to develop a Retrieval Augmented Generation (RAG) system using an open-source LLM with LangChain. The RAG will be built using a vector publications database, as provided. The model and development is expected to retrieve, summarize and generate relevant research findings on Autism, Therapy, and Intervention based on a user query.


Certainly! Let's go through the process of creating a RAG (Retrieval-Augmented Generation) model using Hugging Face and integrating it with an LLM model like "mistralai/Mistral-7B-Instruct-v0.2" using LangChain. We'll also discuss the role of LangChain in the code for application process integration.

## First, let's install the necessary libraries:
```
!pip install -q langchain
!pip install sentence-transformers
!pip install -q torch
!pip install -q faiss-gpu
!pip install -q -U langchain transformers bitsandbytes accelerate


import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import BitsAndBytesConfig
from langchain import PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
```
## extracting text from pdf files
```
# Function to extract text from PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()
    return text

# Function to remove figures, tables, and in-text citations from text
def clean_text(text):
    # Remove figures and tables
    text = re.sub(r'\[[0-9]*\]', '', text)
    text = re.sub(r'\[[a-z]*\]', '', text)

    # Remove in-text citations (assuming citations are in the form of [Author, Year])
    text = re.sub(r'\[\w+\,\s\d+\]', '', text)

    return text

# Function to tokenize text into sentences
def tokenize_text(text):
    sentences = sent_tokenize(text)
    return sentences

# Main function to process PDF files
def process_pdf(pdf_file):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_file)

    # Clean text
    cleaned_text = clean_text(text)

    # Tokenize text into sentences
    sentences = tokenize_text(cleaned_text)

    # Save clean text to a new file
    with open(pdf_file.replace('.pdf', '_clean.txt'), 'w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write(sentence + '\n')

# Process each research paper PDF file
pdf_files = os.listdir('/content/drive/MyDrive/Publications')

for pdf_file in pdf_files:
  file_path = os.path.join('/content/drive/MyDrive/Publications', pdf_file )
  process_pdf(file_path)
```
## Preprocessing the text files
```
# Preprocessing the text files by removing stop words, stemming, and lemmatizing.
# Removeing citations and references from the retrieved text.

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Apply stemming and lemmatization
    stemmed_words = [stemmer.stem(word) for word in words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]

    # Join the processed words back into a single string
    processed_text = ' '.join(lemmatized_words)

    return processed_text

def remove_citations_and_references(text):
    # Remove in-text citations (e.g., (Author, Year))
    text = re.sub(r'\([A-Za-z]+,\s\d{4}\)', '', text)

    # Remove references (e.g., [1], [2], [3])
    text = re.sub(r'\[\d+\]', '', text)

    return text

def preprocess_and_clean_text_files(directory):
    cleaned_texts = []

    # Loop through each file in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)

            # Read the contents of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Remove citations and references
            text = remove_citations_and_references(text)

            # Preprocess the text
            preprocessed_text = preprocess_text(text)

            # Append the preprocessed text to the list
            cleaned_texts.append(preprocessed_text)

    return cleaned_texts

text_files_directory = "/content/drive/MyDrive/Publications/clean files"

# Preprocess and clean text files
cleaned_texts = preprocess_and_clean_text_files(text_files_directory)

```


# Creating splites
```
def split_text_files(text_list, chunk_size, chunk_overlap):
    # Instantiate the RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


    # List to store all chunks from the input text list
    all_chunks = []


    # Process each text in the input list
    for text in text_list:
        # Split the text into chunks using the splitter
        chunks = splitter.split_text(text)


        # Append the chunks to the list of all chunks
        all_chunks.extend(chunks)






    return all_chunks
```




# Split text files into chunks
```
chunks = split_text_files(cleaned_texts, chunk_size=512, chunk_overlap=0.2)

```
```

# Model is being prepared to create embeddings
# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"


# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cuda'}


# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}


# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

```
HuggingFaceEmbeddings: LangChain integrates with Hugging Face's embedding models to create vector representations of the loaded documents. We create an instance of HuggingFaceEmbeddings to generate embeddings for our documents.


## Vector database
```

db = FAISS.from_texts(chunks, embeddings)
```

FAISS: LangChain provides a vector store implementation using FAISS (Facebook AI Similarity Search) to efficiently store and retrieve documents based on their embeddings. We create a FAISS vector store from the loaded documents and their embeddings.

## Load the LLM model and tokenizer
```
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_4bit = AutoModelForCausalLM.from_pretrained( "mistralai/Mistral-7B-Instruct-v0.2", device_map="auto",quantization_config=quantization_config)

```
```
# Create a tokenizer object by loading the pretrained "mistralai/Mistral-7B-Instruct-v0.2" tokenizer.
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
```

## Creating instance of the HuggingFacePipeline
```
question_answerer = pipeline(
    "text-generation",
    model=model_4bit,
    tokenizer=tokenizer,
    max_length=8192,
    # max_new_tokens=512,
    # return_tensors='pt',
    return_text=True
)
```
```
# Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
# with additional model-specific arguments (temperature and max_length)
llm = HuggingFacePipeline(
    pipeline=question_answerer,
    model_kwargs = {"max_length": 8192}
)
```

Create a retriever object from the 'db' using the 'as_retriever' method..
```
retriever = db.as_retriever()
```

We create a retriever using the vector store index, which will be used to retrieve relevant documents for a given query.

```
# Create a retriever object from the 'db' with a search configuration where it retrieves up to 2 relevant splits/documents.
retriever = db.as_retriever(search_kwargs={"k": 2})
```

```
# Create a question-answering instance (qa) using the RetrievalQA class.
# It's configured with a language model (llm), a chain type "refine," the retriever we created, and an option to not return source documents.
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=False)

```

We create a RetrievalQA chain using the LLM and the retriever. This chain combines the retrieval and question-answering capabilities.

## Now you can use the pipeline to get answers for questions
question = "What are the variety of Multimodal and Multi-modular AI Approaches to Streamline Autism Diagnosis in Young Children?"
```
result = qa.run({"query": question})
print(result)
```








