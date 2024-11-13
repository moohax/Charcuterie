import json
import typer
import sys
from rich import print
from dotenv import load_dotenv
import os
import rigging as rg
from typing import Optional
import asyncio

API_KEYS = {
    "OPENAI_API_KEY": None,
    "ANTHROPIC_API_KEY": None,
    "MISTRAL_API_KEY": None,
    "TOGETHER_API_KEY": None,
    "TOGETHERAI_API_KEY": None
}

app = typer.Typer()

@app.command()
def history():
    try:
        with open("prompts.json", "r") as f:
            data = json.load(f)
            for prompt in data.get("prompts", []):
                print(prompt)
    except FileNotFoundError:
        print("\n[!] No history found (prompts.json doesn't exist)\n")
    except json.JSONDecodeError:
        print("\n[!] Error reading history file\n")

@app.command()
def create(
    model: str = typer.Option("gpt-3.5-turbo", "--model", help="Model to use. Examples: gpt-3.5-turbo, claude-3-sonnet, mistral-medium"),
    temp: float = typer.Option(0.7, "--temperature", help="Controls randomness"),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Maximum response length"),
    top_p: int = typer.Option(1, "--top-p", help="Controls response diversity"),
    frequency_penalty: int = typer.Option(0, "--freq-pen", help="Reduces repetition"),
    presence_penalty: int = typer.Option(0, "--pres-pen", help="Encourages new topics"),
    save: bool = typer.Option(False, "--save", help="Save conversation to prompts.json")):

    async def run_chat():
        prompt = typer.prompt("\nModel Input")

        # get the appropriate API key based on model
        api_key = None
        if "gpt" in model.lower() or "openai" in model.lower():
            api_key = API_KEYS["OPENAI_API_KEY"]
        elif "claude" in model.lower():
            api_key = API_KEYS["ANTHROPIC_API_KEY"]
        elif "mistral" in model.lower():
            api_key = API_KEYS["MISTRAL_API_KEY"]
        elif "together" in model.lower():
            api_key = API_KEYS["TOGETHER_API_KEY"] or API_KEYS["TOGETHERAI_API_KEY"]

        # create generator with API key if available
        generator_id = f"{model},temperature={temp}"
        if api_key:
            generator_id += f",api_key={api_key}"

        generator = rg.get_generator(generator_id)

        # build chat pipeline with user's prompt
        pipeline = generator.chat([
            {"role": "user", "content": prompt}
        ]).with_(
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )

        try:
            chat = await pipeline.run()
            response_text = chat.last.content

            entry = {
                "input": prompt,
                "parameters": {
                    "model": model,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty
                },
                "output": [response_text]
            }

            if save:
                with open("prompts.json", "r+") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = {"prompts": []}

                    data["prompts"].append(entry)
                    f.seek(0)
                    json.dump(data, f, indent=4)

            print(entry)

        except Exception as e:
            print(f"\n[!] Error: {str(e)}\n")
            sys.exit(1)

    asyncio.run(run_chat())

if __name__ == "__main__":
    load_dotenv()

    for key in API_KEYS:
        API_KEYS[key] = os.getenv(key)

    if not any(API_KEYS.values()):
        print("\n[!] Warning: No API keys found in environment. Set at least one of:")
        for key in API_KEYS:
            print(f"  - {key}")
        print("\n")
        sys.exit(0)

    app()