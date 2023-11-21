from tensorflow import keras
from tensorflow.keras import layers
from functools import partial
from .layer import TimeSelectionLayer, binary_sigmoid_unit, TimeSelectionLayerConstant
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
import numpy as np

class MultiBoostingRegressor:
    def __init__(self, n_estimators=100, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []

    def fit(self, X, y):
        residual = y.copy()
        for i in range(self.n_estimators):
            estimator = DecisionTreeRegressor(max_depth=3)

            estimator.fit(X, residual)
            self.estimators.append(estimator)

            # Actualizar el residual para el siguiente estimador
            predictions = estimator.predict(X)
            residual -= predictions

    def predict(self, X):
        # Inicializar predicciones a cero
        y_pred = np.zeros((X.shape[0], self.estimators[0].n_outputs_))
        for estimator in self.estimators:
            y_pred += estimator.predict(X)
        return y_pred
    
class ParallelBoostingRegressor:
    def __init__(self, n_estimators=100, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = None
        self.n_targets = None

    def fit(self, X, y):
        # Determinar el número de targets basado en la forma de y
        if y.ndim == 1:
            y = y[:, np.newaxis]
            self.n_targets = 1
        else:
            self.n_targets = y.shape[1]

        # Inicializar una lista de listas para los estimadores
        self.estimators = [[] for _ in range(self.n_targets)]

        for target in range(self.n_targets):
            residual = y[:, target].copy()
            for i in range(self.n_estimators):
                estimator = DecisionTreeRegressor(max_depth=self.max_depth)

                estimator.fit(X, residual)
                self.estimators[target].append(estimator)

                # Actualizar el residual para el siguiente estimador
                predictions = estimator.predict(X)
                residual -= predictions

    def predict(self, X):
        # Inicializar predicciones a cero
        y_pred = np.zeros((X.shape[0], self.n_targets))
        for target in range(self.n_targets):
            for estimator in self.estimators[target]:
                y_pred[:, target] += estimator.predict(X)
        return y_pred


def get_hyperparameters() -> tuple:
    """
    Get hyperparameters for the model.

    Returns:
        tuple: A tuple containing loss and metrics.
    """

    loss = keras.losses.MSE
    metrics = [keras.metrics.MSE, keras.metrics.MAE,
               keras.metrics.mean_absolute_percentage_error]

    return loss, metrics


def get_base_layer(layer_type: str) -> callable:
    """
    Get the base layer function based on the layer type.

    Args:
        layer_type (str): The type of layer ('dense', 'lstm', or 'cnn').

    Returns:
        callable: The base layer function.
    """
    if layer_type == 'dense':
        layer_base = layers.Dense
    elif layer_type == 'lstm':
        layer_base = layers.LSTM
    elif layer_type == 'cnn':
        layer_base = partial(layers.Conv1D, kernel_size=3)

    return layer_base

def head_layers(parameters: dict, n_features_out: int, name: str = '') -> list:
    """
    Create head layers based on the selection type in the parameters.

    Args:
        parameters (dict): The model parameters.
        n_features_out (int): Number of output features.
        name (str, optional): Name for the layers. Defaults to ''.

    Returns:
        list: List of head layers.
    """
    selection = parameters['selection']['name']
    select_timesteps = parameters['dataset']['params'].get('select_timesteps', True)
    
    head_layers = []
    if selection == 'TimeSelectionLayer':
        regularization = parameters['selection']['params']['regularization']
        head_layers.append(TimeSelectionLayer(num_outputs=n_features_out,
                           regularization=regularization, name=f'{name}'))

    elif selection == 'TimeSelectionLayerConstant':
        regularization = parameters['selection']['params']['regularization']
        head_layers.append(TimeSelectionLayerConstant(num_outputs=n_features_out,
                           regularization=regularization, name=f'{name}'))
    
    if parameters['model']['name'] == 'dense':
        head_layers.append(layers.Flatten())
    
    if len(head_layers)>0:
        return head_layers
    else:
        return None
    

def get_tf_model(parameters: dict, label_idxs: list, values_idxs: list) -> keras.Model:
    """
    Create a TensorFlow model based on the given parameters.

    Args:
        parameters (dict): The model parameters.
        label_idxs (list): List of label indices.
        values_idxs (list): List of value indices.

    Returns:
        keras.Model: The TensorFlow model.
    """
    model = parameters['model']['name']
    n_layers = parameters['model']['params']['layers']
    n_units = parameters['model']['params']['units']
    dropout = parameters['model']['params']['dropout']
    lr = parameters['model']['params']['lr']
    pred_len = parameters['dataset']['params']['pred_len']
    seq_len = parameters['dataset']['params']['seq_len']
    
    loss, metrics = get_hyperparameters()

    n_features_in = len(label_idxs) + len(values_idxs)
    n_features_out = len(label_idxs)
        
    layer_base = get_base_layer(model)
    
    inputs_raw = layers.Input(shape=(seq_len*n_features_in,), name='inputs')
    inputs = layers.Reshape((seq_len, n_features_in), name='inputs_reshaped')(inputs_raw)
    
    header = keras.Sequential(head_layers(parameters, n_features_out*pred_len, name=f'selection_in'))
    
    x = inputs if header is None else header(inputs)
    
    for i in range(n_layers):

        if model == 'lstm' and i < n_layers-1:
            kargs = {"return_sequences": True}
        else:
            kargs = {}

        x = layer_base(n_units, activation="relu" if model != 'lstm' else "tanh", name=f"layer{i}", **kargs)(x)
        x = layers.Dropout(dropout)(x)
        
    outputs = layers.Dense(n_features_out*pred_len, name="output")(x)
    model = keras.Model(inputs=inputs_raw, outputs=outputs, name="tsmodel")
        
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics
    )
    
    return model


def get_sk_model(parameters: dict):
    """
    Create a scikit-learn model based on the given parameters.

    Args:
        parameters (dict): The model parameters.

    Returns:
        object: The scikit-learn model.
    """

    model = parameters['model']['name']

    if model == 'decisiontree':
        model = DecisionTreeRegressor(max_depth=parameters['model']['params']['max_depth'])
    elif model == 'lasso':
        model = Lasso(alpha=parameters['model']['params']['regularization'])
    elif model == 'randomforest':
        model = RandomForestRegressor(max_depth=parameters['model']['params']['max_depth'], n_estimators=parameters['model']['params']['n_estimators'])
    elif model == 'gradientboosting':
        model = GradientBoostingRegressor(max_depth=parameters['model']['params']['max_depth'], n_estimators=parameters['model']['params']['n_estimators'])
    elif model == 'multi_boosting':
        model = MultiBoostingRegressor(max_depth=parameters['model']['params']['max_depth'], n_estimators=parameters['model']['params']['n_estimators'])
    elif model == 'parallel_boosting':
        model = ParallelBoostingRegressor(max_depth=parameters['model']['params']['max_depth'], n_estimators=parameters['model']['params']['n_estimators'])
    else:
        raise NotImplementedError()

    return model


def get_model(parameters: dict, label_idxs: list, values_idxs: list):
    """
    Create a model based on the given parameters.

    Args:
        parameters (dict): The model parameters.
        label_idxs (list): List of label indices.
        values_idxs (list): List of value indices.

    Returns:
        object: The model.
    """

    model_type = parameters['model']['params']['type']

    if model_type == 'tensorflow':
        model = get_tf_model(parameters, label_idxs, values_idxs)
    else:
        model = get_sk_model(parameters)

    return model


def get_selected_idxs(model: keras.Model, features: np.ndarray) -> set:
    """
    Get selected indices from the model's selection layers.

    Args:
        model (keras.Model): The TensorFlow model.
        features (np.ndarray): Input features.

    Returns:
        set: Set of selected indices.
    """
    
    selected_idxs = set()
    for layer in model.layers:
        if 'selection' in layer.name:
            mask = binary_sigmoid_unit(layer.get_mask()).numpy()
            selected_idxs = selected_idxs.union(np.arange(0, features.flatten().shape[0])[
                mask.flatten().astype(bool)].tolist())
        elif type(layer) == keras.Sequential:
            selected_idxs = selected_idxs.union(get_selected_idxs(layer, features))
    return selected_idxs
