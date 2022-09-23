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
