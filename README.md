# YouTube Transcript Q&A App

This application allows users to ask questions about YouTube video content by analyzing the video's transcript using RAG (Retrieval-Augmented Generation) technology.

## Features

- Extract transcripts from YouTube videos
- Process and analyze video content
- Answer questions about the video content using AI
- User-friendly Gradio interface

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Hugging Face token as an environment variable:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```

## Usage

1. Run the application:
   ```bash
   python youtube_transcript_rag.py
   ```
2. Open the provided Gradio interface in your browser
3. Enter a YouTube video ID (the part after `v=` in the YouTube URL)
4. Type your question about the video content
5. Get AI-generated answers based on the video transcript

## Example

For a video with URL `https://www.youtube.com/watch?v=JaRGJVrJBQ8`, the video ID would be `JaRGJVrJBQ8`.

## Requirements

- Python 3.7+
- Hugging Face account and API token
- Internet connection for accessing YouTube videos

## Note

This application requires a Hugging Face token to access the language models. Make sure to set up your token before running the application. 