import onnx
import onnxruntime as ort
import numpy as np
import threading
import asyncio

class WLActor:
    def __init__(self, model_path):
        # model
        self._model = onnx.load(model_path)
        self._session = ort.InferenceSession(model_path)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # buffers
        self._num_observations = 19
        self._num_actions = 4
        self._num_latents = 3
        self._observation = np.zeros((self._num_observations + self._num_latents,), dtype=np.float32)
        self._action = np.zeros((self._num_actions,), dtype=np.float32)
        self._latent = np.zeros((self._num_latents,), dtype=np.float32) # output from encoder

        # threading
        self._ready = np.full((5,), False, dtype=bool)
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run)

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

    def _ready_to_run(self):
        return self._ready.all()
    
    def input_base_ang_vel(self, base_ang_vel):
        '''base_ang_vel: [x, y, z]'''
        with self._lock:
            self._observation[:3] = base_ang_vel
            self._ready[0] = True

    def input_projected_gravity(self, projected_gravity):
        '''projected_gravity: [x, y, z]'''
        with self._lock:
            self._observation[3:6] = projected_gravity
            self._ready[1] = True

    def input_commands(self, commands):
        '''commands: [lin_vel_x, lin_vel_y, ang_vel_yaw]'''
        with self._lock:
            self._observation[6:9] = commands
            self._ready[2] = True

    def input_dof_pos(self, dof_pos):
        '''dof_pos: [left_leg_pos, right_leg_pos]'''
        with self._lock:
            self._observation[9:11] = dof_pos
            self._ready[3] = True

    def input_dof_vel(self, dof_vel):
        '''dof_vel: [left_leg_vel, right_leg_vel, left_wheel_vel, right_wheel_vel]'''
        with self._lock:
            self._observation[11:15] = dof_vel
            self._ready[4] = True

    def output_action(self) -> np.ndarray:
            '''action: [l_leg_pos, r_leg_pos, l_wheel_vel, r_wheel_vel]'''
            with self._lock:
                return self._action

    async def _step(self):
        async with self._lock:
            self._observation[15:19] = self._action
            self._observation[19:22] = self._latent
            # Run session.run in a thread pool to avoid blocking the asyncio loop
            self._action = await self._loop.run_in_executor(None, lambda: self._session.run([self._output_name], {self._input_name: self._observation})[0])

    async def _run(self):
        '''Run the actor. Main loop.'''
        while not self._shutdown.is_set():
            if self._ready_to_run():
                await self._step()
            else:
                await asyncio.sleep(0.005)  # Non-blocking wait
