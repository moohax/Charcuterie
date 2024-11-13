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
        # initialize empty list to store all entries
        entries = []

        with open("prompts.jsonl", "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                messages = entry.get("messages", [])
                generated = entry.get("generated", [])

                timestamp = entry.get("timestamp", "N/A")
                if timestamp != "N/A":
                    # Convert to more readable format: YYYY-MM-DD HH:MM:SS UTC
                    timestamp = timestamp.replace('T', ' ').split('.')[0] + " UTC"

                entries.append({
                    "timestamp": timestamp,
                    "input": messages[0].get("content", "N/A") if messages else "N/A",
                    "parameters": {
                        "model": entry.get("generator_id", "N/A").split(",")[0],
                        "temperature": entry.get("generator_id", "N/A").split("temperature=")[1].split(",")[0] if "temperature=" in entry.get("generator_id", "") else "N/A"
                    },
                    "output": [generated[0].get("content", "N/A") if generated else "N/A"]
                })

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Timestamp", style="cyan")
        table.add_column("Input", style="green")
        table.add_column("Model", style="yellow")
        table.add_column("Output", style="blue")

        for entry in entries:
            output_text = entry["output"][0]
            if len(output_text) > 100:
                output_text = output_text[:100] + "..."

            table.add_row(
                entry["timestamp"],
                entry["input"],
                entry["parameters"]["model"],
                output_text
            )

        console.print("\n[bold]Chat History[/bold]")
        console.print(table)

    except FileNotFoundError:
        console.print("\n[red][!] No history found (prompts.jsonl doesn't exist)[/red]\n")
    except json.JSONDecodeError as e:
        console.print(f"\n[red][!] Error reading history file: {str(e)}[/red]\n")
    except Exception as e:
        console.print(f"\n[red][!] Unexpected error: {str(e)}[/red]\n")

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
        try:
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

            # create pipeline with watch callback if save is enabled
            pipeline = generator.chat([
                {"role": "user", "content": prompt}
            ]).with_(
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )

            if save:
                log_to_file = rg.watchers.write_chats_to_jsonl("prompts.jsonl")
                pipeline = pipeline.watch(log_to_file)

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
            console.print(output_panel)

            if save:
                console.print("\n[green][+] Successfully saved to prompts.jsonl[/green]")

        except Exception as e:
            console.print(f"\n[red][!] Error: {str(e)}[/red]")
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