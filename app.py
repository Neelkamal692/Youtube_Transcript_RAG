from dotenv import load_dotenv
load_dotenv()

import os

if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError("Please set the GOOGLE_API_KEY environment variable with your Google API key.")

import gradio as gr
from supadata import Supadata
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize Supadata
supadata = Supadata(api_key=os.environ.get("SUPADATA_API_KEY"))

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separators=["\n\n", "\n", " ", ".", ","],
    length_function=len,
    is_separator_regex=False
)

# Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Initialize the LLM
# llm = HuggingFaceEndpoint(
#     repo_id="microsoft/Phi-3-mini-4k-instruct",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
#     huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN")
# )

chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=5000,
    timeout=None,
    max_retries=2
)

# Define the prompt template
prompt = PromptTemplate(
    template="""
You are an intelligent AI assistant specialized in analyzing YouTube video transcripts. Your task is to provide accurate, detailed, and helpful answers based solely on the provided transcript content.

IMPORTANT GUIDELINES:
- Answer ONLY from the provided transcript context
- If the context is insufficient to answer the question, clearly state "I don't have enough information from the transcript to answer this question"
- Provide specific details and examples from the transcript when possible
- Be concise but comprehensive in your responses
- If asked for a summary, organize the information logically
- If asked about specific topics, focus on what was actually discussed in the video
- Maintain a helpful and informative tone

TRANSCRIPT CONTEXT:
{context}

QUESTION: {question}

Please provide a clear and detailed answer based on the transcript above:
""",
    input_variables=['context', 'question']
)

# Global variable to store the current retriever
current_retriever = None
current_video_id = None

def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    if "youtube.com/watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url  # Assume it's already a video ID

def process_video_url(video_url_or_id):
    """Process video URL and create retriever object."""
    global current_retriever, current_video_id
    
    try:
        # Extract video ID if URL is provided
        video_id = extract_video_id(video_url_or_id)
        
        # Check if we already have a retriever for this video
        if current_video_id == video_id and current_retriever is not None:
            return f"✅ Video already processed: {video_id}"
        
        # Get transcript using Supadata
        transcript_response = supadata.youtube.transcript(
            video_id=video_id,
            text=True  # Get plain text transcript
        )
        
        # Extract the transcript text
        full_transcript_text = transcript_response.content
        # print(f"✅ Full transcript text: {full_transcript_text}")
        
        # Create chunks
        chunks = text_splitter.create_documents([full_transcript_text])
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        current_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})
        # print(f"✅ Current Retreiver : {current_retriever}")
        current_video_id = video_id
        
        return f"✅ Video processed successfully: {video_id}"
    
    except Exception as e:
        return f"❌ Error processing video: {str(e)}"

def answer_question(question):
    global current_retriever
    
    if current_retriever is None:
        return "❌ Process a video first."
    
    try:
        # Retrieve docs with scores
        docs_and_scores = current_retriever.vectorstore.similarity_search_with_score(
            question, k=8
        )
        for doc, score in docs_and_scores:
            print(f"[DEBUG] score={score:.3f}, text={doc.page_content}")
        
        # Extract docs
        retrieved_docs = [doc for doc, _ in docs_and_scores]
        
        # Build context and reply as before
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        # print("\nContext text:\n", context_text)
        final_prompt = prompt.invoke({"context": context_text, "question": question})
        answer = chat.invoke(final_prompt)
        return answer.content
    
    except Exception as e:
        print(f"[ERROR] in answer_question: {e}")
        return f"❌ Error: {str(e)}"

def process_video(video_url_or_id, question):
    """Legacy function for backward compatibility."""
    # Process video first
    process_result = process_video_url(video_url_or_id)
    if "❌" in process_result:
        return process_result
    
    # Then answer question
    return answer_question(question)

def main():
    with gr.Blocks(title="YouTube Transcript Q&A", theme=gr.themes.Soft()) as iface:
        gr.Markdown("""
        # YouTube Transcript Q&A
        
        Ask questions about any YouTube video's content! 
        
        ## How to use:
        1. Paste a YouTube video URL (e.g., https://www.youtube.com/watch?v=JaRGJVrJBQ8) or just the video ID
        2. Click "Process Video" to download and process the transcript (this happens once per video)
        3. Type your question about the video content
        4. Click "Ask Question" to get your answer
        """)
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Textbox(
                    label="YouTube Video URL or ID",
                    placeholder="Enter YouTube URL or video ID (e.g., JaRGJVrJBQ8)",
                    lines=1
                )
                process_btn = gr.Button("Process Video", variant="primary")
                process_status = gr.Textbox(
                    label="Processing Status",
                    lines=1,
                    interactive=False
                )
                
                gr.Markdown("---")
                
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What would you like to know about this video?",
                    lines=2
                )
                ask_btn = gr.Button("Ask Question", variant="secondary")
            
            with gr.Column():
                output = gr.Textbox(
                    label="Answer",
                    lines=8,
                    show_copy_button=True
                )
        
        # Example inputs
        gr.Examples(
            examples=[
                ["https://www.youtube.com/watch?v=-moW9jvvMr4&t=1s", "What is this video about?"],
                ["https://www.youtube.com/watch?v=-moW9jvvMr4&t=1s", "What are the main topics discussed in this video?"],
                ["https://www.youtube.com/watch?v=-moW9jvvMr4&t=1s", "Can you summarize the key points from this video?"],
                ["-moW9jvvMr4", "What are the most important takeaways from this content?"]
            ],
            inputs=[video_input, question_input]
        )
        
        # Connect the buttons
        process_btn.click(
            fn=process_video_url,
            inputs=[video_input],
            outputs=[process_status]
        )
        
        ask_btn.click(
            fn=answer_question,
            inputs=[question_input],
            outputs=[output]
        )
        
        iface.launch()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        result = process_video("https://www.youtube.com/watch?v=gN-QWM5iY9M", "What is this video about?")
        print(result)
    else:
        main()