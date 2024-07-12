import onnx
import onnxruntime as ort
import numpy as np
import threading
import asyncio

class ObsEncoder:
    def __init__(self, model_path):
        self._session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_names = [input.name for input in self._session.get_inputs()]
        print('Encoder input names:', input_names)
        output_names = [output.name for output in self._session.get_outputs()]
        print('Encoder output names:', output_names)
        self._input_name = input_names[0]
        self._output_name = output_names[0]

        # buffers
        self._num_observations = 19 * 5
        self._observation_history = np.zeros((self._num_observations,), dtype=np.float32)

    def step(self, observation):
        self._observation_history[:-19] = self._observation_history[19:]
        self._observation_history[-19:] = observation
        input_data = {self._input_name: self._observation_history}
        latent = self._session.run([self._output_name], input_data)[0]
        return latent