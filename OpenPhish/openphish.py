import json
import typer
import inspect
import openai
import sys
from rich import print

app = typer.Typer()

@app.command()
def engines():
    engines = openai.Engine.list()
    print(engines)


@app.command()
def history():
    with open("prompts.json", "r+") as f:
        data = json.load(f)

    for prompt in data.get("prompts"):
        print(
            {
                "prompt": prompt["input"],
                "output": prompt["output"]
            }
        )

@app.command()
def create(
    model: str = typer.Option("text-davinci-002", "--model", help="model to use"),
    temp: float = typer.Option(0.7, "--temperature", help=""),
    max_tokens: int = typer.Option(256, "--max-tokens", help=""),
    top_p: int = typer.Option(1, "--top-p", help=""),
    frequency_penalty: int = typer.Option(0, "--freq-pen", help=""),
    presence_penalty: int = typer.Option(0, "--pres-pen", help=""),
    save: bool = typer.Option(False, "--save", help="")):
    
    prompt = typer.prompt("\nModel Input")

    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    entry = {
        "input": prompt,
        "parameters": "",
        "output": []
    }

    for text in response["choices"]:
        entry["output"].append(text["text"])


    with open("prompts.json", "r+") as f:
        data = json.load(f)
        data["prompts"].append(entry)
        
        f.seek(0)
        # convert back to json.
        json.dump(data, f, indent = 4)

    print(entry)
    

if __name__ == "__main__":
    openai.api_key =  ""
    if not openai.api_key:
        print("\n[!] Grab a key from beta.openai.com\n")
        sys.exit(0)

    app()