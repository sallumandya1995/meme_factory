import gradio as gr
import json
import re
import random
from openai import OpenAI
import fal_client
from pydantic import BaseModel
import os
from huggingface_hub import HfApi

# Environment variables from Hugging Face Secrets
api_key = os.environ.get('API_KEY')
api_base = os.environ.get('API_BASE')
FAL_KEY = os.environ.get('FAL_KEY')

# Initialize OpenAI client
client = OpenAI(
    api_key=api_key,
    base_url=api_base
)

model = "google/gemini-flash-1.5"

class MemeRequest(BaseModel):
    domain: str

class MemeResponse(BaseModel):
    image_url: str
    top_text: str
    bottom_text: str

def generate_meme(domain: str) -> MemeResponse:
    """
    Generates a meme based on the given domain.
    """
    temperature = round(random.uniform(0.5, 0.9), 2)
    print(f"Temperature: {temperature}")

    user_content = (f"Act like Non offensive meme maker. Always create different and unique funny memes "
                   f"always remember stable diffusion cannot render text natively hence, write prompt just detailing the scene or image without text "
                   f"Return meme details in below json format for topic {domain}\n\n"
                   "{ \"stableDiffusionPrompt\": \" \", \"topText\": \"\", \"bottomText\": \"\" }\n\n"
                   "Strictly just reply json no extra text")

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful meme maker, who makes non offensive memes(Dont include Cat in memes)"},
                {"role": "user", "content": user_content}
            ],
            model=model,
            temperature=temperature
        )

        result = chat_completion.choices[0].message.content

        try:
            meme_data = json.loads(result)
        except json.JSONDecodeError:
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                json_string = json_match.group(1)
                meme_data = json.loads(json_string)
            else:
                raise ValueError("No valid JSON found in the response")

        print("Stable Diffusion Prompt:", meme_data['stableDiffusionPrompt'])
        print("Top Text:", meme_data['topText'])
        print("Bottom Text:", meme_data['bottomText'])

        # Configure fal_client with API key
        fal_client.api_key = FAL_KEY

        handler = fal_client.submit(
            "fal-ai/flux/schnell", 
            arguments={
                "prompt": meme_data['stableDiffusionPrompt'],
                "image_size": "landscape_4_3",
                "num_inference_steps": 4,
                "num_images": 1,
                "enable_safety_checker": True
            } 
        )

        result = handler.get()
        image_url = result['images'][0]['url']
        print("Generated Image URL:", image_url)

        return MemeResponse(
            top_text=meme_data['topText'],
            image_url=image_url, 
            bottom_text=meme_data['bottomText']
        )

    except Exception as e:
        print(f"Error generating meme: {str(e)}")
        raise

def generate_meme_gradio(domain):
    """
    Wrapper function for Gradio interface.
    """
    try:
        meme_response = generate_meme(domain)
        return meme_response.top_text, meme_response.image_url, meme_response.bottom_text
    except Exception as e:
        return "Error", None, f"Failed to generate meme: {str(e)}"

# Example domains for the interface
example_domains = [
    'HR', 'Technology', 'AI', 'Machine Learning', 'Sales', 'Marketing',
    'Remote Work', 'Coffee', 'Monday', 'Deadlines', 'Office Pranks', 'Tech Support',
    'Social Media', 'Startup Life', 'Zoom Fails', 'Work-Life Balance', 'Coding',
    'Data Privacy', 'Cybersecurity', 'Cloud Computing', 'Blockchain', 'IoT',
    'Virtual Reality', 'Augmented Reality', 'Quantum Computing', '5G', 'AI Ethics'
]

# Create Gradio interface
demo = gr.Interface(
    fn=generate_meme_gradio,
    inputs=[
        gr.Textbox(
            label="Enter your meme topic",
            value="Human Resource team",
            placeholder="Type a topic or choose from examples below",
            info="Try topics like 'AI', 'Remote Work', or 'Tech Support'"
        )
    ],
    outputs=[
        gr.Textbox(label="Top Text"),
        gr.Image(label="Generated Meme"),
        gr.Textbox(label="Bottom Text")
    ],
    theme=gr.themes.Ocean(),
    title="🎨 AI Meme Factory",
    description="""
    Welcome to the AI Meme Factory! World's First Generative meme Engine.
    Generate unique, funny, and non-offensive memes using the power of AI.

    🤖 Powered by:
    - Gemini for creative text generation
    - Flux for high-quality image generation

    ✨ Features:
    - Instant meme generation
    - Professional and work-appropriate content
    - Wide range of topics supported
    """,
    examples=[[domain] for domain in random.sample(example_domains, 6)],
    allow_flagging="never",
    cache_examples=False
)

# Launch the app
if __name__ == "__main__":
    demo.launch(debug=True)
