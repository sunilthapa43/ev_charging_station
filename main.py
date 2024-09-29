from utils.evaluation import evaluate_tiny_ml
from utils.get_mlp_model import get_model, get_tinyml_model

# get the MLP model

model, X_test, y_test = get_model()

# now get the tiny_ml model and evaluate it if it exists else convert the mlp model to tinyml
interpreter, X_test, y_test = get_tinyml_model()

# evaluate the tiny_ml_model
evaluate_tiny_ml(interpreter, X_test, y_test)

