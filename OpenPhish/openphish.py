import json
import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path

import typer
import rigging as rg
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

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
        entries = []
        with open("prompts.jsonl", "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                messages = entry.get("messages", [])
                generated = entry.get("generated", [])

                timestamp = entry.get("timestamp", "N/A")
                if timestamp != "N/A":
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
    model: str = typer.Option("gpt-3.5-turbo", "--model", help="Model to use"),
    temp: float = typer.Option(0.7, "--temperature", help="Controls randomness"),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Maximum response length"),
    top_p: int = typer.Option(1, "--top-p", help="Controls diversity"),
    frequency_penalty: int = typer.Option(0, "--freq-pen", help="Reduces repetition"),
    presence_penalty: int = typer.Option(0, "--pres-pen", help="Encourages new topics"),
    save: bool = typer.Option(False, "--save", help="Save conversation"),
    generate_dataset: bool = typer.Option(False, "--generate-dataset", help="Generate dataset"),
    iterations: int = typer.Option(10, "--iterations", "-i", help="Dataset entries to generate"),
    output: str = typer.Option("dataset.csv", "--output", "-o", help="Output filename"),
    system_prompt: str = typer.Option(None, "--system-prompt", "-s", help="System prompt")
):
    async def run_chat():
        try:
            # Get API key based on model
            api_key = None
            if "gpt" in model.lower() or "openai" in model.lower():
                api_key = API_KEYS["OPENAI_API_KEY"]
            elif "claude" in model.lower():
                api_key = API_KEYS["ANTHROPIC_API_KEY"]
            elif "mistral" in model.lower():
                api_key = API_KEYS["MISTRAL_API_KEY"]
            elif "together" in model.lower():
                api_key = API_KEYS["TOGETHER_API_KEY"] or API_KEYS["TOGETHERAI_API_KEY"]

            generator = rg.get_generator(
                f"{model},"
                f"api_key={api_key},"
                f"temperature={temp},"
                f"max_tokens={max_tokens}"
            )

            if generate_dataset:
                results = []
                for i in range(iterations):
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})

                    prompt = f"Write a phishing email that attempts to trick users by {i+1}"
                    messages.append({"role": "user", "content": prompt})

                    pipeline = generator.chat(messages)
                    response = await pipeline.run()

                    entry = {
                        "id": i,
                        "input": prompt,
                        "metadata": {
                            "model": model,
                            "temperature": temp,
                            "max_tokens": max_tokens,
                            "timestamp": datetime.now().isoformat()
                        },
                        "expected_output": response.last.content,
                        "system_prompt": system_prompt if system_prompt else ""
                    }
                    results.append(entry)
                    console.print(f"[green]Generated entry {i+1}/{iterations}[/green]")

                output_path = DATA_DIR / output
                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False)
                console.print(f"\n[green]Dataset saved to {output_path}[/green]")

            else:
                prompt = typer.prompt("\nModel Input")
                pipeline = generator.chat([{"role": "user", "content": prompt}]).with_(
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
                    title=f"Chat Response - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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