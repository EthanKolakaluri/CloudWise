import gradio as gr
import json
import requests
from langchain_community.embeddings import OllamaEmbeddings
from chromadb.config import Settings
from chromadb import Client
from langchain.vectorstores import Chroma

# Initialize Ollama embeddings using DeepSeek-R1
try:
    embedding_function = OllamaEmbeddings(model="deepseek-r1:1.5b")
    print("Ollama embeddings initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Ollama embeddings: {e}")
    embedding_function = None

# Initialize Chroma client
try:
    client = Client(Settings())
    print("Chroma client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Chroma client: {e}")
    client = None

# Check if the collection exists before deleting it
if client:
    try:
        client.delete_collection(name="cloud_vms")  # Delete existing collection (if any)
        print("Existing collection 'cloud_vms' deleted.")
    except ValueError as e:
        if "does not exist" in str(e):  # Handle the case where the collection doesn't exist
            print("Collection 'cloud_vms' does not exist. Skipping deletion.")
        else:
            print(f"Error deleting collection: {e}")

# Create the collection
if client:
    try:
        collection = client.create_collection(name="cloud_vms")
        print("Collection 'cloud_vms' created successfully.")
    except Exception as e:
        print(f"Failed to create collection: {e}")
        collection = None

# Function to load JSON data from a file
def load_json_from_file(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Failed to load JSON data: {e}")
        return {"VMs": []}  # Fallback to empty data

# Function to add JSON data to Chroma
def add_json_to_chroma(json_data):
    if not collection:
        print("Chroma collection not initialized. Skipping data addition.")
        return

    # Extract VM details from JSON
    vms = json_data.get("VMs", [])
    for idx, vm in enumerate(vms):
        # Create a text representation of the VM details
        vm_text = (
            f"VM Name: {vm['VM Name']}, "
            f"Cost: {vm['Cost']}, "
            f"CPU Usage: {vm['CPU Usage (%)']}%, "
            f"Memory Usage: {vm['Memory Usage (%)']}%, "
            f"Total Usage: {vm['Total Usage (Hours)']} hours, "
            f"Instance Type: {vm['Instance Type']}, "
            f"Disk Size: {vm['Disk Size (GB)']} GB, "
            f"OS Type: {vm['OS Type']}, "
            f"Region: {vm['Region']}"
        )
        # Generate embedding for the VM text
        if embedding_function:
            try:
                embedding = embedding_function.embed_query(vm_text)
                # Add to Chroma collection
                collection.add(
                    documents=[vm_text],  # Use the VM details as the document
                    metadatas=[vm],  # Store the entire VM object in metadata
                    embeddings=[embedding],  # Embedding for the VM details
                    ids=[str(idx)]  # Unique ID for each VM
                )
                print(f"Added VM {idx + 1} to Chroma collection.")
            except Exception as e:
                print(f"Failed to add VM {idx + 1} to Chroma collection: {e}")
        else:
            print("Embedding function not initialized. Skipping VM addition.")

# Load JSON data from a file
json_file_path = "dataset1.json"  # Replace with the path to your JSON file
json_data = load_json_from_file(json_file_path)

# Add the JSON data to Chroma
add_json_to_chroma(json_data)

# Initialize retriever using Ollama embeddings for queries
if client and embedding_function:
    try:
        retriever = Chroma(collection_name="cloud_vms", client=client, embedding_function=embedding_function).as_retriever()
        print("Retriever initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        retriever = None
else:
    retriever = None

def retrieve_context(question):
    if not retriever:
        return "Retriever not initialized."

    # Retrieve relevant VMs based on the question
    try:
        results = retriever.invoke(question)
        # Combine the retrieved VM details
        context = "\n\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        print(f"Failed to retrieve context: {e}")
        return ""

def query_deepseek(question, context):
    if not embedding_function:
        return "Embedding function not initialized."

    # Format the input prompt
    formatted_prompt = f"Question: {question}\n\nContext: {context}"

    # Query Ollama using HTTP request (replace with actual API endpoint)
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",  # Ollama's API endpoint for generating responses
            headers={"Content-Type": "application/json"},
            json={
                "model": "deepseek-r1:1.5b",
                "prompt": formatted_prompt,
                "stream": False  # Set to False to get a single response
            }
        )
        
        # Check if the response is successful
        if response.status_code == 200:
            response_data = response.json()
            # Extract and return the response content
            return response_data.get("response", "No response content")
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        print(f"Failed to query DeepSeek: {e}")
        return "Failed to generate an answer."

# Introduction Page
def introduction_page():
    return gr.Markdown("""
    # Welcome to CloudWise

    ## Introduction
    CloudWise is a powerful tool designed to help you optimize and customize your cloud infrastructure. Whether you're looking to reduce costs, improve performance, or find the best instance types for your workloads, CloudWise has you covered.

    ### Key Features
    - **VM Optimization**: Get recommendations for the best VMs based on your product description.
    - **Cost Analysis**: Identify the most expensive VMs and learn how to reduce costs.
    - **Resource Usage**: Monitor CPU and memory usage to ensure optimal performance.
    - **Instance Recommendations**: Find the best instance types for high-performance databases or other workloads.
    - **Region Analysis**: Discover which regions offer the lowest costs for running your VMs.
    - **Disk Size Optimization**: Learn how to optimize disk sizes for your VMs.
    - **OS Type Recommendations**: Find the most cost-effective OS types for your workloads.

    ### How to Use
    Navigate to the **Cloudbot** tab to start asking questions and get real-time recommendations. Use the example prompts provided to get started quickly.

    ### Example Prompts
    - **VM optimization based on product description**: "What VM's would be best to use for (product description)?"
    - **Cost Optimization**: "Which VMs are the most expensive and how can I reduce their costs?"
    - **Resource Usage**: "Show me VMs with high CPU or memory usage."
    - **Instance Recommendations**: "What instance type is best for a high-performance database?"
    - **Region Analysis**: "Which region has the lowest cost for running my VMs?"
    - **Disk Size**: "How can I optimize disk sizes for my VMs?"
    - **OS Type**: "What are the most cost-effective OS types for my workloads?"

    <br><br>
    Feel free to explore and ask any question related to optimizing or customizing your cloud infrastructure!
    """)

# Chatbot Interface
def chatbot_interface():
    custom_css = """
        /* Set the main background color to white */
        .gradio-container {
            background-color: #FFFFFF !important;
        }

        /* Style the container around the textboxes */
        .gradio-row {
            background-color: #FFFFFF !important;  /* White background */
            padding: 20px !important;  /* Add some padding */
            border-radius: 10px !important;  /* Rounded corners */
            margin: 10px 0 !important;  /* Add some margin */
        }

        /* Style the textboxes */
        .gradio-textbox {
            background-color: #FFFFFF !important;  /* White background for textboxes */
            border: 1px solid #CCCCCC !important;  /* Light gray border */
            border-radius: 5px !important;
            color: #000000 !important;  /* Black text */
        }

        /* Style the labels */
        .gradio-label {
            color: #000000 !important;  /* Black text for labels */
            font-weight: bold !important;
        }

        /* Style the submit button */
        .gradio-button {
            background-color: #1E90FF !important;  /* Dodger blue button */
            color: #FFFFFF !important;  /* White text */
            border: none !important;
            border-radius: 5px !important;
        }

        /* Hover effect for the button */
        .gradio-button:hover {
            background-color: #187BCD !important;  /* Slightly darker blue on hover */
        }
    """
    with gr.Blocks(css=custom_css) as demo:
        # Title at the top
        gr.Markdown("""
        <h1 style='text-align: center; font-size: 50px;'>
            <span style='color: #63B8FF;'>Cloud</span><span style='
                color: #000000; /* Black text for "Wise" */
                text-shadow: 
                    -1px -1px 0 #FFFFFF,  
                     1px -1px 0 #FFFFFF,
                    -1px  1px 0 #FFFFFF,
                     1px  1px 0 #FFFFFF; /* White border */
            '>Wise</span>
        </h1>
        """)
        
        gr.Markdown("""
        ### Example Prompts
        Here are some examples of prompts you can use to interact with CloudWise:
        
        - **VM optimization based on product description**: "What VM's would be best to use for (product description)?"
        - **Cost Optimization**: "Which VMs are the most expensive and how can I reduce their costs?"
        - **Resource Usage**: "Show me VMs with high CPU or memory usage."
        - **Instance Recommendations**: "What instance type is best for a high-performance database?"
        - **Region Analysis**: "Which region has the lowest cost for running my VMs?"
        - **Disk Size**: "How can I optimize disk sizes for my VMs?"
        - **OS Type**: "What are the most cost-effective OS types for my workloads?"
        
        <br><br>
        Feel free to ask any question related to optimizing or customizing your cloud infrastructure!
        """)

        # Create text input for user question and output text area
        question_input = gr.Textbox(label="Ask any question to optimize or customize your cloud", placeholder="Enter your question here...")
        answer_output = gr.Textbox(label="Answer", interactive=False)
        
        # Define the interaction logic
        def get_answer(question):
            context = retrieve_context(question)
            answer = query_deepseek(question, context)
            return answer
        
        question_input.submit(get_answer, inputs=question_input, outputs=answer_output)

    return demo

# Create the tabbed interface
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Introduction"):
            introduction_page()
        with gr.TabItem("Cloudbot"):
            chatbot_interface()

# Launch the app with Gradio Blocks
demo.launch()
