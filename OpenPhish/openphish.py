import json
import typer
import sys
from rich import print
from dotenv import load_dotenv
import os
import rigging as rg
from typing import Optional
import asyncio
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

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

            table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
            table.add_column("Timestamp", style="cyan")
            table.add_column("Input", style="green")
            table.add_column("Model", style="yellow")
            table.add_column("Output", style="blue")

            for prompt in data.get("prompts", []):
                # Handle both old and new format
                timestamp = prompt.get("timestamp", "N/A")

                # Handle parameters being either string or dict
                params = prompt.get("parameters", {})
                model = params.get("model", "N/A") if isinstance(params, dict) else "N/A"

                output_text = prompt["output"][0] if prompt["output"] else "N/A"
                if len(output_text) > 100:
                    output_text = output_text[:100] + "..."

                table.add_row(
                    timestamp,
                    prompt["input"],
                    model,
                    output_text
                )

            console.print("\n[bold]Chat History[/bold]")
            console.print(table)

    except FileNotFoundError:
        console.print("\n[red][!] No history found (prompts.json doesn't exist)[/red]\n")
    except json.JSONDecodeError:
        console.print("\n[red][!] Error reading history file[/red]\n")

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
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

            output_panel = Panel(
                f"[bold green]Input:[/bold green] {prompt}\n\n"
                f"[bold yellow]Parameters:[/bold yellow]\n"
                f"  Model: {model}\n"
                f"  Temperature: {temp}\n"
                f"  Max Tokens: {max_tokens}\n"
                f"  Top P: {top_p}\n"
                f"  Frequency Penalty: {frequency_penalty}\n"
                f"  Presence Penalty: {presence_penalty}\n\n"
                f"[bold cyan]Response:[/bold cyan]\n{response_text}",
                title=f"Chat Response - {entry['timestamp']}",
                box=box.ROUNDED,
                padding=(1, 2)
            )

            if save:
                try:
                    try:
                        with open("prompts.json", "r") as f:
                            data = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        data = {"prompts": []}

                    data["prompts"].append(entry)

                    with open("prompts.json", "w") as f:
                        json.dump(data, f, indent=4)

                    console.print("\n[green][+] Saved to prompts.json[/green]")

                except Exception as e:
                    console.print(f"\n[red][!] Error saving to prompts.json: {str(e)}[/red]")

            console.print(output_panel)

        except Exception as e:
            console.print(f"\n[red][!] Error: {str(e)}[/red]\n")
            sys.exit(1)

    asyncio.run(run_chat())

if __name__ == "__main__":
    load_dotenv()

    for key in API_KEYS:
        API_KEYS[key] = os.getenv(key)

    if not any(API_KEYS.values()):
        console.print("\n[red][!] Warning: No API keys found in environment. Set at least one of:[/red]")
        for key in API_KEYS:
            console.print(f"  - {key}")
        console.print("\n")
        sys.exit(0)

    app()