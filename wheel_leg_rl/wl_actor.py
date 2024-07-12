import onnx
import onnxruntime as ort
import numpy as np
import threading
import asyncio
from wheel_leg_rl.obs_encoder import ObsEncoder

class WLActor:
    def __init__(self, actor_path, encoder_path, dt=0.005):
        # model
        self._session = ort.InferenceSession(actor_path, providers=['CPUExecutionProvider'])
        input_names = [input.name for input in self._session.get_inputs()]
        print('Actor input names:', input_names)
        output_names = [output.name for output in self._session.get_outputs()]
        print('Actor output names:', output_names)
        self._input_name = input_names[0]
        self._output_name = output_names[0]

        self._encoder = ObsEncoder(encoder_path)

        # buffers
        self._num_observations = 19
        self._num_actions = 4
        self._num_latents = 3
        self._observation = np.zeros((self._num_observations,), dtype=np.float32)
        self._action = np.zeros((self._num_actions,), dtype=np.float32)
        self._latent = np.zeros((self._num_latents,), dtype=np.float32) # output from encoder

        # threading
        self._dt = dt
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_async_loop)

    def __del__(self):
        self.stop()

    def _start_async_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run())
        self._loop.close()

    def start(self):
        self._thread.start()

    def stop(self):
        self._shutdown.set()
        self._thread.join()
    
    def input_base_ang_vel(self, base_ang_vel):
        '''base_ang_vel: [x, y, z]'''
        with self._lock:
            self._observation[:3] = base_ang_vel

    def input_projected_gravity(self, projected_gravity):
        '''projected_gravity: [x, y, z]'''
        with self._lock:
            self._observation[3:6] = projected_gravity

    def input_commands(self, commands):
        '''commands: [lin_vel_x, lin_vel_y, ang_vel_yaw]'''
        with self._lock:
            self._observation[6:9] = commands

    def input_dof_pos(self, dof_pos):
        '''dof_pos: [left_leg_pos, right_leg_pos]'''
        with self._lock:
            self._observation[9:11] = dof_pos

    def input_dof_vel(self, dof_vel):
        '''dof_vel: [left_leg_vel, right_leg_vel, left_wheel_vel, right_wheel_vel]'''
        with self._lock:
            self._observation[11:15] = dof_vel

    def output_action(self) -> np.ndarray:
            '''action: [l_leg_pos, r_leg_pos, l_wheel_vel, r_wheel_vel]'''
            with self._lock:
                return self._action

    async def _run(self):
        while not self._shutdown.is_set():
            with self._lock:
                observation = self._observation.copy()
            latent = self._encoder.step(observation)
            input_data = {self._input_name: np.concatenate((observation, latent))}
            action = self._session.run([self._output_name], input_data)[0]
            with self._lock:
                self._action = action
            await asyncio.sleep(self._dt)