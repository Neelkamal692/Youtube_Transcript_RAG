from dotenv import load_dotenv
load_dotenv()

import os
if not os.getenv("HUGGINGFACE_TOKEN"):
    raise RuntimeError("Please set the HUGGINGFACE_TOKEN environment variable with your Hugging Face API token.")
else:
    print(os.getenv("HUGGINGFACE_TOKEN"))

import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ".", ","],
    length_function=len,
    is_separator_regex=False
)

# Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN")
)

chat = ChatHuggingFace(llm=llm, verbose=True)

# Define the prompt template
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=['context', 'question']
)

def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    if "youtube.com/watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url  # Assume it's already a video ID

def process_video(video_url_or_id, question):
    try:
        # Extract video ID if URL is provided
        video_id = extract_video_id(video_url_or_id)
        
        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Extract text segments
        list_of_text_segments = [item['text'] for item in transcript]
        full_transcript_text = " ".join(list_of_text_segments)
        
        # Create chunks
        chunks = text_splitter.create_documents([full_transcript_text])
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # Generate answer
        final_prompt = prompt.invoke({"context": context_text, "question": question})
        answer = chat.invoke(final_prompt)
        
        return answer.content
    
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    with gr.Blocks(title="YouTube Transcript Q&A", theme=gr.themes.Soft()) as iface:
        gr.Markdown("""
        # YouTube Transcript Q&A
        
        Ask questions about any YouTube video's content! Just paste the video URL or ID and your question.
        
        ## How to use:
        1. Paste a YouTube video URL (e.g., https://www.youtube.com/watch?v=JaRGJVrJBQ8) or just the video ID
        2. Type your question about the video content
        3. Click Submit to get your answer
        """)
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Textbox(
                    label="YouTube Video URL or ID",
                    placeholder="Enter YouTube URL or video ID (e.g., JaRGJVrJBQ8)",
                    lines=1
                )
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What would you like to know about this video?",
                    lines=2
                )
                submit_btn = gr.Button("Submit", variant="primary")
            
            with gr.Column():
                output = gr.Textbox(
                    label="Answer",
                    lines=5,
                    show_copy_button=True
                )
        
        # Example inputs
        gr.Examples(
            examples=[
                ["https://www.youtube.com/watch?v=JaRGJVrJBQ8", "What is this video about?"],
                ["JaRGJVrJBQ8", "What are the main topics discussed?"],
                ["https://www.youtube.com/watch?v=JaRGJVrJBQ8", "Summarize the key points"]
            ],
            inputs=[video_input, question_input]
        )
        
        submit_btn.click(
            fn=process_video,
            inputs=[video_input, question_input],
            outputs=output
        )
        iface.launch()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        result = process_video("https://www.youtube.com/watch?v=gN-QWM5iY9M", "What is this video about?")
        print(result)
    else:
        main()