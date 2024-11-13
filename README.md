# Charcuterie - A little bit of everything.
This is a collection of code execution techniques for ML or ML adjacent libraries and a sample attack on a blackbox model using Optuna.

## Quick Start
```
git clone https://github.com/moohax/Charcuterie.git
cd Charcuterie
pip install -r requirements.txt
python charcuterie.py --help
```

## Available techniques
```
jupyer-auto-load                               Jupyter autoload via %html
numpy-array                                    Loads code through a numpy.asarray() call by implementing the __array__() method required by NumPy.
numpy-load                                     Standard numpy.load()
numpy-load-library                             Loads a dll, so, or dylib via numpy.ctypeslib.load_library()
onnx-convert-ort                               Loads code via a custom_op_library during conversion from ONNX to the internal ORT model format.
onnx-session-options                           Loads code via ONNX SessionOptions.register_custom_ops().
optimize-attack                                Runs Optuna against the "discovered" number of inputs for the toy model
pandas-read-csv                                Uses Pandas default behavoir to read a local file via fsspec
pandas-read-pickle                             Standard pandas.read_pickle()
pickle-load                                    Standard pickle.load()
sklearn-load                                   Standard Sklearn joblib.load()
tf-dll-hijack                                  Writes a dll to search path prior to Tensorflow import.
tf-load-library                                Loads an op library, dll, so, or dylib via tf.load_library()
tf-load-op-library                             Loads an op library, dll, so, or dylib via tf.load_op_library()
torch-classes-load-library                     Loads a dll, so, or dylib via torch.classes.load_library()
torch-jit                                      Load code via torch.jit.load()
torch-load                                     Standard torch.load()
```

# OpenPhish
Use any model to generate phishy materials supported using [rigging](https://rigging.dreadnode.io/).

## Quick Start
```
Get an API key from OpenAI and set an environment variable (`OPENAI_API_KEY=your-api-key-here`)
python ./openphish.py create
python ./openphish.py history
```

experiment with optional parameters:

- `--model gpt-3.5-turbo` - Select the model to use
- `--temperature 0.8` - Control response creativity (0.0-1.0)
- `--max-tokens 256` - Maximum length of response
- `--top-p 1` - Control response diversity
- `--freq-pen 0` - Reduce repetition in responses
- `--pres-pen 0` - Encourage new topics
- `--save` - Save the conversation to prompts.json

for models, [rigging](https://rigging.dreadnode.io/) uses identifier strings in this format:

`<provider>!<model>,<**kwargs>`

examples:

```
"gpt-3.5-turbo,temperature=0.5"
"openai/gpt-4,api_key=sk-1234"
"litellm!claude-3-sonnet-2024022"
"anthropic/claude-2.1,stop=output:;---,seed=1337"
"together_ai/meta-llama/Llama-3-70b-chat-hf"
```