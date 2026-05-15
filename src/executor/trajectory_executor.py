import json
import numpy as np
import time
import airsim
import os
import sys
import cv2
import random
import threading
import copy
import socket
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import threading

try:
    from tornado.iostream import StreamClosedError
    TORNADO_AVAILABLE = True
except ImportError:
    class StreamClosedError(Exception):
        pass
    TORNADO_AVAILABLE = False

tqdm.set_lock(threading.RLock())

def safe_log(msg, scene_id=None):
    if scene_id:
        msg = f"[{scene_id}] {msg}"
    tqdm.write(msg, file=sys.stderr)

try:
    import msgpackrpc
    MSGPACKRPC_AVAILABLE = True
except ImportError:
    MSGPACKRPC_AVAILABLE = False
    print("Warning: msgpackrpc is not installed; auto scene startup will be unavailable")

try:
    import logging
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    root_logger.propagate = False
    
    from .logger import logger
    if hasattr(logger, 'logger'):
        for handler in logger.logger.handlers[:]:
            logger.logger.removeHandler(handler)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        logger.logger.propagate = False
    USE_LOGGER = True
except:
    class SimpleLogger:
        @staticmethod
        def info(msg): print(f"[INFO] {msg}")
        @staticmethod
        def warning(msg): print(f"[WARNING] {msg}")
        @staticmethod
        def error(msg): print(f"[ERROR] {msg}")
    logger = SimpleLogger()
    USE_LOGGER = False


class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.flag_ok = False

    def run(self):
        self.result = self.func(*self.args)
        self.flag_ok = True

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except:
            return None


class AirVLNSimulatorClientTool:
    def __init__(self, machines_info) -> None:
        if not MSGPACKRPC_AVAILABLE:
            raise RuntimeError("msgpackrpc is not installed; auto scene startup is unavailable")

        self.machines_info = copy.deepcopy(machines_info)
        self.socket_clients = []
        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in machines_info]
        self.airsim_ports = []
        self.airsim_ip = '127.0.0.1'
        self._init_check()
        self.objects_name_cnt = [[0 for _ in list(item['open_scenes'])] for item in machines_info]

    def _init_check(self) -> None:
        ips = [item['MACHINE_IP'] for item in self.machines_info]
        assert len(ips) == len(set(ips)), 'MACHINE_IP repeat'

    def _confirmSocketConnection(self, socket_client) -> bool:
        try:
            socket_client.call('ping')
            print("Connected\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            return True
        except:
            try:
                print("Ping returned false\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            except:
                print('Ping returned false')
            return False

    def _confirmConnection(self) -> bool:
        all_confirmed = True
        max_retries = 10
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    confirmed = False
                    count = 0
                    if USE_LOGGER:
                        logger.info(f"Start confirming connection: clients[{index_1}][{index_2}]")
                    while not confirmed and count < max_retries:
                        try:
                            self.airsim_clients[index_1][index_2].ping()
                            confirmed = True
                            if USE_LOGGER:
                                logger.info(f"Connection confirmed: clients[{index_1}][{index_2}]")
                        except Exception as e:
                            if count < 3 or count % 5 == 0:
                                if USE_LOGGER:
                                    logger.warning(f"Connection confirm failed (attempt {count + 1}): {str(e)}")
                            count += 1
                            if count >= max_retries:
                                if USE_LOGGER:
                                    logger.error(f"Connection confirm failed after {max_retries} retries: clients[{index_1}][{index_2}]")
                                all_confirmed = False
        
        return all_confirmed

    def _closeSocketConnection(self) -> None:
        socket_clients = self.socket_clients
        for socket_client in socket_clients:
            try:
                socket_client.close()
            except Exception as e:
                pass
        self.socket_clients = []
        return

    def _closeConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    try:
                        self.airsim_clients[index_1][index_2].close()
                    except Exception as e:
                        pass
        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in self.machines_info]
        return

    def run_call(self, airsim_timeout: int = 180) -> None:
        socket_clients = []
        for index, item in enumerate(self.machines_info):
            socket_clients.append(
                msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=600)
            )

        for socket_client in socket_clients:
            if not self._confirmSocketConnection(socket_client):
                logger.error('cannot establish socket')
                raise Exception('cannot establish socket')

        self.socket_clients = socket_clients

        before = time.time()
        self._closeConnection()

        def _run_command(index, socket_client: msgpackrpc.Client):
            logger.info(f"Start opening scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}")
            logger.info(f'gpus: {self.machines_info[index]}')
            result = socket_client.call('reopen_scenes', socket_client.address._host, list(zip(self.machines_info[index]['open_scenes'], self.machines_info[index]['gpus'])))
            if result[0] == False:
                error_detail = result[1] if len(result) > 1 and result[1] is not None else 'No error detail'
                logger.error(f"Failed to open scenes, machine: {socket_client.address._host}:{socket_client.address._port}")
                logger.error(f"Error detail: {error_detail}")
                raise Exception(f"Failed to open scenes: {error_detail}")
            assert len(result[1]) == 2, "Failed to open scenes"
            wait_time = 3 * len(self.machines_info[index]['open_scenes']) + 35
            if USE_LOGGER:
                logger.info("Waiting for scenes to start...")
            else:
                print(f'waiting for airsim connection...')
            ip = result[1][0]
            if isinstance(ip, bytes):
                ip = ip.decode('utf-8')
            ports = result[1][1]
            self.airsim_ip = ip
            self.airsim_ports = ports
            assert str(ip) == str(socket_client.address._host), "Failed to open scenes"
            assert len(ports) == len(self.machines_info[index]['open_scenes']), "Failed to open scenes"
            for i, port in enumerate(ports):
                if self.machines_info[index]['open_scenes'][i] is None:
                    self.airsim_clients[index][i] = None
                else:
                    self.airsim_clients[index][i] = airsim.MultirotorClient(ip=ip, port=port, timeout_value=airsim_timeout)
                    if not USE_LOGGER:
                        print(f"AirSim client port: {port}")

            if USE_LOGGER:
                logger.info(f"Scenes opened, machine {index}: {socket_client.address._host}:{socket_client.address._port}")
            
            max_wait_time = 180
            check_interval = 5
            waited = 0
            ready = False

            port_check_interval = 2
            port_check_count = 0
            max_port_checks = 150

            if USE_LOGGER:
                logger.info(f"Waiting for port {ports[0]} to start listening...")
            port_listening = False
            scene_exited = False
            log_file_pattern = None
            try:
                from pathlib import Path
                log_dir = Path(__file__).parent.parent / 'core' / 'logs'
                if log_dir.exists() and len(self.machines_info[index]['open_scenes']) > 0:
                    scene_id = self.machines_info[index]['open_scenes'][0]
                    if scene_id is not None:
                        log_file_path = log_dir / f'scene_{scene_id}_{ports[0]}.log'
                        if log_file_path.exists():
                            log_file_pattern = log_file_path
                        else:
                            log_files = list(log_dir.glob(f'scene_*_{ports[0]}.log'))
                            if log_files:
                                log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                                log_file_pattern = log_files[0]
            except:
                pass
            
            while port_check_count < max_port_checks:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex((ip, ports[0]))
                    sock.close()
                    if result == 0:
                        if USE_LOGGER:
                            logger.info(f"Port {ports[0]} is listening")
                        port_listening = True
                        break
                except Exception as e:
                    pass
                
                if log_file_pattern and log_file_pattern.exists() and port_check_count > 0 and port_check_count % 10 == 0:
                    try:
                        with open(log_file_pattern, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            if lines:
                                last_lines = ''.join(lines[-5:])
                                if 'Exiting abnormally' in last_lines or 'Exiting.' in last_lines or 'LogExit: Exiting' in last_lines:
                                    scene_exited = True
                                    if USE_LOGGER:
                                        logger.error(f"Detected scene exit (from log file {log_file_pattern.name})")
                                    break
                    except:
                        pass
                
                port_check_count += 1
                if port_check_count % 5 == 0 and not USE_LOGGER:
                    print(f"Waiting for port {ports[0]} to listen... ({port_check_count * port_check_interval}s)")

                time.sleep(port_check_interval)
            
            if not port_listening:
                if scene_exited:
                    error_msg = f"Scene exited during startup (port {ports[0]} never started listening)"
                    if log_file_pattern and log_file_pattern.exists():
                        error_msg += f", check log file: {log_file_pattern}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                process_status = "unknown"
                try:
                    process_status = "port not listening; scene may have crashed or hung"
                except:
                    pass

                error_msg = f"Port {ports[0]} did not start listening within {max_port_checks * port_check_interval} seconds; scene may have crashed. {process_status}"
                if log_file_pattern and log_file_pattern.exists():
                    error_msg += f", check log file: {log_file_pattern}"
                logger.error(error_msg)
                logger.error("Check the scene log file for more details")
                raise Exception(error_msg)

            while waited < max_wait_time:
                try:
                    self.airsim_clients[index][0].ping()
                    objs = self.airsim_clients[index][0].simListSceneObjects()
                    if objs and len(objs) > 0:
                        if USE_LOGGER:
                            logger.info(f"Scene loaded. objects={len(objs)}")
                        ready = True
                        break
                    else:
                        if not USE_LOGGER and waited % 10 == 0:
                            print(f"Connected but scene objects list is empty; waiting... {waited}/{max_wait_time}s")
                except Exception as e:
                    error_msg = str(e)
                    if not USE_LOGGER and waited % 10 == 0:
                        if "ECONNREFUSED" in error_msg or "Connection refused" in error_msg:
                            print(f"Scene not ready; waiting... {waited}/{max_wait_time}s (connection refused)")
                        else:
                            print(f"Scene not ready; waiting... {waited}/{max_wait_time}s")
                waited += check_interval

            if not ready:
                logger.warning(f"Scene readiness check timed out ({max_wait_time}s); it may still be initializing.")
            
            return ports

        threads = []
        thread_results = []
        for index, socket_client in enumerate(socket_clients):
            threads.append(
                MyThread(_run_command, (index, socket_client))
            )
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            thread.get_result()
            thread_results.append(thread.flag_ok)
        threads = []
        
        if not (np.array(thread_results) == True).all():
            raise Exception('Failed to start one or more AirSim scene threads')

        after = time.time()
        diff = after - before
        if USE_LOGGER:
            logger.info(f"Scene startup completed in {diff:.2f}s")
        else:
            print(f"Scene startup completed in {diff:.2f}s")

        assert self._confirmConnection(), 'server connect failed'
        self._closeSocketConnection()



class TrajectoryExecutor:
    
    def __init__(self, 
                 scene_id="env_400",
                 sim_server_host="127.0.0.1",
                 sim_server_port=30000,
                 gpu_id=0,
                 scene_index=1,
                 uav_vehicle_name="Drone_1",
                 target_object_name="UAV1",
                 target_asset_name=None,
                 target_object_scale=(1.0, 1.0, 1.0),
                 camera_name="0",
                 auto_start_scene=True,
                 pre_existing_client=None, 
                 pre_existing_sim_client_tool=None,
                 deterministic_step_mode=True):
        self.scene_id = scene_id
        self.sim_server_host = sim_server_host
        self.sim_server_port = sim_server_port
        self.gpu_id = gpu_id
        self.scene_index = scene_index
        self.uav_vehicle_name = uav_vehicle_name
        self.target_object_name = target_object_name
        self.target_asset_name = target_asset_name if target_asset_name is not None else target_object_name
        self._target_asset_name_explicitly_set = (target_asset_name is not None)
        self.target_object_scale = target_object_scale
        self.camera_name = camera_name
        self.auto_start_scene = auto_start_scene
        self.pre_existing_client = pre_existing_client
        self.pre_existing_sim_client_tool = pre_existing_sim_client_tool
        self.deterministic_step_mode = bool(deterministic_step_mode)
        self._sim_paused_by_executor = False

        
        self.client = None
        self.sim_client_tool = None
        self._connected_scene_id = None
        
        self._prev_frame_data = None
        
        self._abnormal_jumps = []
    

    def _safe_sim_pause(self, pause: bool):
        try:
            self.client.simPause(bool(pause))
            self._sim_paused_by_executor = bool(pause)
        except Exception:
            pass

    def _safe_continue_for_frames(self, frames: int):
        if frames is None:
            return
        frames = int(frames)
        if frames <= 0:
            return
        try:
            self._safe_sim_pause(True)
            self.client.simContinueForFrames(frames)
            self._safe_sim_pause(True)
        except Exception:
            import time
            try:
                self._safe_sim_pause(False)
                time.sleep(0.02 * frames)
            finally:
                self._safe_sim_pause(True)

    def _get_vehicle_pos(self):
        pose = self.client.simGetVehiclePose(vehicle_name=self.uav_vehicle_name)
        p = pose.position
        return float(p.x_val), float(p.y_val), float(p.z_val)

    def _set_vehicle_pose_paused(self, x, y, z, quat, retries=3, tol_xy=0.3, tol_z=1.0):
        import time
        import numpy as np
        import airsim

        x, y, z = float(x), float(y), float(z)

        self._safe_sim_pause(True)
        last = None
        for k in range(int(retries)):
            pose = airsim.Pose(airsim.Vector3r(x, y, z), quat)
            rpc_ok = False
            last_rpc_err = None
            for rpc_try in range(5):
                try:
                    self.client.simSetVehiclePose(
                        pose,
                        ignore_collision=True,
                        vehicle_name=self.uav_vehicle_name,
                    )
                    rpc_ok = True
                    break
                except Exception as e:
                    last_rpc_err = e
                    err_l = str(e).lower()
                    if (
                        rpc_try < 4
                        and ("timeout" in err_l or "timed out" in err_l or "request timed out" in err_l)
                    ):
                        time.sleep(0.6 * (rpc_try + 1))
                        continue
                    raise
            if not rpc_ok and last_rpc_err is not None:
                raise last_rpc_err

            try:
                self.client.simContinueForFrames(2)
            except Exception:
                pass

            px, py, pz = self._get_vehicle_pos()
            last = np.array([px, py, pz], dtype=np.float32)

            err_xy = float(np.hypot(px - x, py - y))
            err_z = float(abs(pz - z))
            if (err_xy <= float(tol_xy)) and (err_z <= float(tol_z)):
                return True, last, float(np.linalg.norm(last - np.array([x, y, z], dtype=np.float32))), err_xy, err_z

        err = float(np.linalg.norm(last - np.array([x, y, z], dtype=np.float32))) if last is not None else float("inf")
        err_xy = float(np.hypot(last[0] - x, last[1] - y)) if last is not None else float("inf")
        err_z = float(abs(last[2] - z)) if last is not None else float("inf")
        return False, last, err, err_xy, err_z

    def _set_object_pose_paused(self, object_name, x, y, z, quat=None, retries=3, tol=1.0):
        import numpy as np
        import airsim

        x, y, z = float(x), float(y), float(z)
        if quat is None:
            quat = airsim.to_quaternion(0, 0, 0)

        self._safe_sim_pause(True)
        last = None
        for k in range(int(retries)):
            self.client.simSetObjectPose(
                object_name,
                airsim.Pose(airsim.Vector3r(x, y, z), quat)
            )
            pose = self.client.simGetObjectPose(object_name)
            if pose is None:
                continue
            p = pose.position
            last = np.array([float(p.x_val), float(p.y_val), float(p.z_val)], dtype=np.float32)
            if np.any(np.isnan(last)):
                continue
            err = float(np.linalg.norm(last - np.array([x, y, z], dtype=np.float32)))
            if err <= float(tol):
                return True, last, err

        err = float(np.linalg.norm(last - np.array([x, y, z], dtype=np.float32))) if last is not None and not np.any(np.isnan(last)) else float("inf")
        return False, last, err

    def _step_if_needed(self, frames: int = 1):
        if not getattr(self, "deterministic_step_mode", False):
            return
        self._safe_continue_for_frames(frames)

    def connect(self, reuse_connection=True, max_retries=3, retry_delay=2):
        import time
        
        main_scene_id = self.scene_id if isinstance(self.scene_id, str) else self.scene_id[0]
        
        if reuse_connection and self.client is not None and self._connected_scene_id == main_scene_id:
            try:
                self.client.getMultirotorState(vehicle_name=self.uav_vehicle_name)
                if getattr(self, "deterministic_step_mode", False):
                    self._safe_sim_pause(True)
                return self.client
            except:
                print(f"⚠ Existing AirSim client is invalid; reconnecting...")
                self.client = None
                self.sim_client_tool = None
                self._connected_scene_id = None
        
        if self.pre_existing_client is not None:
            self.client = self.pre_existing_client
            if self.pre_existing_sim_client_tool is not None:
                self.sim_client_tool = self.pre_existing_sim_client_tool
            self._connected_scene_id = main_scene_id
            return self.client
        
        if not self.auto_start_scene or not MSGPACKRPC_AVAILABLE:
            raise RuntimeError("AirSim auto-start is unavailable (check auto_start_scene and msgpackrpc)")
        
        last_error = None
        for attempt in range(max_retries):
            try:
                if isinstance(self.scene_id, list) and isinstance(self.gpu_id, list):
                    if len(self.scene_id) != len(self.gpu_id):
                        raise RuntimeError(f"Scene count ({len(self.scene_id)}) does not match GPU count ({len(self.gpu_id)})")
                    open_scenes = self.scene_id
                    gpus = self.gpu_id
                    main_scene_id = open_scenes[0]
                else:
                    open_scenes = [self.scene_id] if not isinstance(self.scene_id, list) else self.scene_id
                    gpus = [self.gpu_id] if not isinstance(self.gpu_id, list) else self.gpu_id
                    main_scene_id = self.scene_id if not isinstance(self.scene_id, list) else self.scene_id[0]
                
                machines_info = [{
                    'MACHINE_IP': self.sim_server_host,
                    'SOCKET_PORT': self.sim_server_port,
                    'open_scenes': open_scenes,
                    'gpus': gpus
                }]
                
                if self.sim_client_tool is not None:
                    try:
                        self.sim_client_tool._closeConnection()
                        self.sim_client_tool._closeSocketConnection()
                    except:
                        pass
                
                if os.environ.get("DAGGER_MULTI_WORKER") != "1":
                    try:
                        tmp_client = msgpackrpc.Client(
                            msgpackrpc.Address(self.sim_server_host, self.sim_server_port),
                            timeout=30,
                        )
                        try:
                            tmp_client.call('close_scenes', self.sim_server_host)
                        except Exception:
                            pass
                        try:
                            tmp_client.close()
                        except Exception:
                            pass
                    except Exception:
                        pass
                    import time as _time
                    _time.sleep(3)
                self.sim_client_tool = AirVLNSimulatorClientTool(machines_info)
                self.sim_client_tool.run_call()
                
                if len(self.sim_client_tool.airsim_clients) > 0 and len(self.sim_client_tool.airsim_clients[0]) > 0:
                    if isinstance(self.scene_id, list):
                        scene_index = open_scenes.index(main_scene_id)
                        self.client = self.sim_client_tool.airsim_clients[0][scene_index]
                    else:
                        self.client = self.sim_client_tool.airsim_clients[0][0]
                    
                    if self.client is None:
                        raise RuntimeError("AirSim client is None")
                    
                    vehicle_names = self.client.listVehicles()
                    if self.uav_vehicle_name not in vehicle_names:
                        raise RuntimeError(f"Vehicle '{self.uav_vehicle_name}' not found. Available vehicles: {vehicle_names}")
                    
                    self.client.getMultirotorState(vehicle_name=self.uav_vehicle_name)
                    max_retries = 5
                    retry_delay = 2
                    for attempt in range(max_retries):
                        try:
                            self.client.enableApiControl(True, vehicle_name=self.uav_vehicle_name)
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                error_msg = str(e)
                                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                                    time.sleep(retry_delay)
                                    continue
                            raise
                    
                    print(f"✓ Connected to AirSim scene: {main_scene_id}")
                    self.client._sim_client_tool = self.sim_client_tool
                    self._connected_scene_id = main_scene_id
                    
                    return self.client
                else:
                    raise RuntimeError("Failed to create AirSim client")
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"⚠ Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                    print(f"  Retrying in {retry_delay}s...")
                    self.client = None
                    if self.sim_client_tool is not None:
                        try:
                            self.sim_client_tool._closeConnection()
                            self.sim_client_tool._closeSocketConnection()
                        except:
                            pass
                        self.sim_client_tool = None
                else:
                    raise RuntimeError(f"Connection failed after {max_retries} retries: {e}")
        
        raise RuntimeError(f"Connection failed after {max_retries} retries: {last_error}")
    
    def disconnect(self):
        self.client = None
        self.sim_client_tool = None
        self._connected_scene_id = None
    
    def load_trajectory(self, json_path):
        json_path = Path(json_path)

        def _as_xyz_list(pos):
            if pos is None:
                return None
            if isinstance(pos, dict) and 'x' in pos and 'y' in pos and 'z' in pos:
                return [pos['x'], pos['y'], pos['z']]
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                return [pos[0], pos[1], pos[2]]
            return None
        
        if json_path.name.endswith('_uav.json') or json_path.name.endswith('_target.json'):
            base_name = json_path.name.replace('_uav.json', '').replace('_target.json', '')
            uav_file = json_path.parent / f"{base_name}_uav.json"
            target_file = json_path.parent / f"{base_name}_target.json"
            
            if not uav_file.exists():
                raise FileNotFoundError(f"UAV trajectory file not found: {uav_file}")
            with open(uav_file, 'r', encoding='utf-8') as f:
                uav_data = json.load(f)
            if 'uav_trajectory' not in uav_data:
                raise ValueError(f"UAV file missing 'uav_trajectory' field")
            uav_traj_data = uav_data['uav_trajectory']
            
            if not target_file.exists():
                raise FileNotFoundError(f"Target trajectory file not found: {target_file}")
            with open(target_file, 'r', encoding='utf-8') as f:
                target_data = json.load(f)
            if 'target_trajectory' not in target_data:
                raise ValueError(f"Target file missing 'target_trajectory' field")
            target_traj_data = target_data['target_trajectory']
            
            is_dataset_format = False
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'trajectory' in data and isinstance(data['trajectory'], list):
                uav_traj_data = []
                for frame in data['trajectory']:
                    if 'uav_position' in frame and frame['uav_position'] is not None:
                        uav_pos_xyz = _as_xyz_list(frame['uav_position'])
                        if uav_pos_xyz is not None:
                            uav_traj_data.append(uav_pos_xyz)
                target_file = json_path.parent / "target_trajectory.json"
                if target_file.exists():
                    with open(target_file, 'r', encoding='utf-8') as f:
                        target_data = json.load(f)
                    if 'target_trajectory_airsim' in target_data and isinstance(target_data['target_trajectory_airsim'], list):
                        target_traj_data = []
                        for p in target_data['target_trajectory_airsim']:
                            if isinstance(p, dict) and 'x' in p and 'y' in p and 'z' in p:
                                target_traj_data.append([p['x'], p['y'], p['z']])
                    else:
                        target_traj_data = []
                else:
                    target_traj_data = []
                if not target_traj_data:
                    for frame in data['trajectory']:
                        if 'target_position' in frame and frame['target_position'] is not None:
                            target_pos_xyz = _as_xyz_list(frame['target_position'])
                            if target_pos_xyz is not None:
                                target_traj_data.append(target_pos_xyz)
                is_dataset_format = True
            elif 'frames' in data and isinstance(data['frames'], list):
                # Compatibility with planner/export format:
                # {"frames":[{"uav_position":[x,y,z], "target_position":[x,y,z], ...}, ...]}
                # NOTE:
                #   The `frames` format here comes from planner/export side data rather than
                #   the saved dataset trajectory format. Its coordinates must therefore be
                #   converted in the same way as raw planner trajectories: y -> -y, z -> -z.
                #   If we treat it as dataset format (only flipping z), the executed y axis
                #   will appear mirrored/reversed in AirSim.
                uav_traj_data = []
                target_traj_data = []
                for frame in data['frames']:
                    if not isinstance(frame, dict):
                        continue
                    uav_pos_xyz = _as_xyz_list(frame.get('uav_position'))
                    target_pos_xyz = _as_xyz_list(frame.get('target_position'))
                    if uav_pos_xyz is not None:
                        uav_traj_data.append(uav_pos_xyz)
                    if target_pos_xyz is not None:
                        target_traj_data.append(target_pos_xyz)
                is_dataset_format = False
            elif 'uav_trajectory' in data and 'target_trajectory' in data:
                uav_traj_data = data['uav_trajectory']
                target_traj_data = data['target_trajectory']
                is_dataset_format = False
            else:
                raise ValueError(f"Input JSON must contain 'uav_trajectory'/'target_trajectory' or 'trajectory'")
        
        if not isinstance(uav_traj_data, list) or len(uav_traj_data) == 0:
            raise ValueError(f"Invalid UAV trajectory data (length={len(uav_traj_data) if isinstance(uav_traj_data, list) else 0}). Check the 'uav_position' field.")
        if not isinstance(target_traj_data, list) or len(target_traj_data) == 0:
            raise ValueError(f"Invalid target trajectory data (length={len(target_traj_data) if isinstance(target_traj_data, list) else 0}). Check the 'target_position' field.")
        
        uav_traj = np.array(uav_traj_data)
        target_traj = np.array(target_traj_data)
        
        if uav_traj.ndim == 0:
            raise ValueError(f"UAV trajectory must be 2D; got a 0D array")
        elif uav_traj.ndim == 1:
            uav_traj = uav_traj.reshape(1, -1)
        
        if target_traj.ndim == 0:
            raise ValueError(f"Target trajectory must be 2D; got a 0D array")
        elif target_traj.ndim == 1:
            target_traj = target_traj.reshape(1, -1)
        
        uav_traj_airsim = np.zeros_like(uav_traj)
        target_traj_airsim = np.zeros_like(target_traj)
        
        if is_dataset_format:
            uav_traj_airsim[:, 0] = uav_traj[:, 0]
            uav_traj_airsim[:, 1] = uav_traj[:, 1]
            uav_traj_airsim[:, 2] = -uav_traj[:, 2]
            
            target_traj_airsim[:, 0] = target_traj[:, 0]
            target_traj_airsim[:, 1] = target_traj[:, 1]
            target_traj_airsim[:, 2] = -target_traj[:, 2]
        else:
            uav_traj_airsim[:, 0] = uav_traj[:, 0]
            uav_traj_airsim[:, 1] = -uav_traj[:, 1]
            uav_traj_airsim[:, 2] = -uav_traj[:, 2]
            
            target_traj_airsim[:, 0] = target_traj[:, 0]
            target_traj_airsim[:, 1] = -target_traj[:, 1]
            target_traj_airsim[:, 2] = -target_traj[:, 2]
        
        return uav_traj_airsim, target_traj_airsim
    
    def _safe_call_airsim(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (StreamClosedError, OSError, ConnectionError) as e:
            return None
        except Exception as e:
            error_msg = str(e).lower()
            if "stream is closed" in error_msg or "connection" in error_msg or "streamclosederror" in error_msg:
                return None
            raise
    
    def reset_collision_info(self):
        try:
            collision_info = self._safe_call_airsim(self.client.simGetCollisionInfo, vehicle_name=self.uav_vehicle_name)
            collision_info = self._safe_call_airsim(self.client.simGetCollisionInfo, vehicle_name=self.uav_vehicle_name)
            self._safe_call_airsim(self.client.simContinueForFrames, 1)
        except Exception as e:
            pass
    

    def teleport_to_start(self, x, y, z, target_x=None, target_y=None, target_z=None, quaternion=None):
        import time
        import numpy as np
        import airsim

        max_teleport_retries = 3

        if quaternion is not None:
            try:
                quat = airsim.Quaternionr(
                    w_val=float(quaternion[0]),
                    x_val=float(quaternion[1]),
                    y_val=float(quaternion[2]),
                    z_val=float(quaternion[3]),
                )
            except Exception:
                quat = airsim.to_quaternion(0, 0, 0)
        else:
            if target_x is not None and target_y is not None:
                yaw = float(np.arctan2(float(target_y) - float(y), float(target_x) - float(x)))
            else:
                yaw = 0.0
            quat = airsim.to_quaternion(0, 0, yaw)

        last = None
        last_err = None

        for attempt in range(max_teleport_retries):
            try:
                try:
                    self.client.enableApiControl(False, vehicle_name=self.uav_vehicle_name)
                except Exception:
                    pass
                try:
                    self.client.armDisarm(False, vehicle_name=self.uav_vehicle_name)
                except Exception:
                    pass

                ok, last, err, err_xy, err_z = self._set_vehicle_pose_paused(
                    x, y, z, quat,
                    retries=1,
                    tol_xy=0.3,
                    tol_z=1.0
                )
                last_err = (err, err_xy, err_z)

                if not ok:
                    if attempt < max_teleport_retries - 1:
                        time.sleep(0.1)
                        continue
                    raise RuntimeError(
                        f"UAV teleport failed after {max_teleport_retries} attempts: "
                        f"target=({float(x):.2f}, {float(y):.2f}, {float(z):.2f}), "
                        f"actual=({last[0]:.2f}, {last[1]:.2f}, {last[2]:.2f}), "
                        f"error={err:.2f}m (XY:{err_xy:.2f}m, Z:{err_z:.2f}m)"
                    )

                max_retries = 5
                retry_delay = 1.0
                for k in range(max_retries):
                    try:
                        self.client.enableApiControl(True, vehicle_name=self.uav_vehicle_name)
                        break
                    except Exception as e:
                        if k < max_retries - 1:
                            msg = str(e).lower()
                            if "timeout" in msg or "timed out" in msg:
                                time.sleep(retry_delay)
                                continue
                        raise

                if getattr(self, "deterministic_step_mode", False):
                    self._step_if_needed(1)
                else:
                    self._safe_sim_pause(False)
                    time.sleep(0.02)

                self.reset_collision_info()

                return

            except Exception:
                if attempt < max_teleport_retries - 1:
                    time.sleep(0.2)
                    continue
                raise

    def spawn_target_object(self, x, y, z):
        try:
            # msgpackrpc cannot serialize numpy scalar types (e.g. numpy.float32).
            # Convert coordinates / scale to plain Python floats before RPC calls.
            try:
                x = float(x)
                y = float(y)
                z = float(z)
            except Exception:
                pass
            try:
                self.client.simDestroyObject(self.target_object_name)
                try:
                    self.client.simContinueForFrames(1)
                except:
                    pass
            except:
                try:
                    pattern = self.target_object_name + ".*"
                    existing_objects = self.client.simListSceneObjects(pattern)
                    for obj_name in existing_objects:
                        try:
                            self.client.simDestroyObject(obj_name)
                        except:
                            pass
                    try:
                        self.client.simContinueForFrames(1)
                    except:
                        pass
                except:
                    pass
            
            pose = airsim.Pose(
                airsim.Vector3r(x, y, z),
                airsim.to_quaternion(0, 0, 0)
            )
            
            sx, sy, sz = self.target_object_scale[0], self.target_object_scale[1], self.target_object_scale[2]
            scale_vector = airsim.Vector3r(float(sx), float(sy), float(sz))
            
            success = self.client.simSpawnObject(
                self.target_object_name,
                self.target_asset_name,
                pose,
                scale_vector,
                physics_enabled=False,
                is_blueprint=False
            )
            
            if success:
                try:
                    self.client.simContinueForFrames(1)
                except:
                    pass
                
                max_verify_attempts = 5
                for verify_attempt in range(max_verify_attempts):
                    try:
                        verify_pose = self.client.simGetObjectPose(self.target_object_name)
                        if verify_pose is not None:
                            actual_pos = np.array([
                                verify_pose.position.x_val,
                                verify_pose.position.y_val,
                                verify_pose.position.z_val
                            ])
                            
                            if np.any(np.isnan(actual_pos)):
                                if verify_attempt < max_verify_attempts - 1:
                                    print(f"  ⚠ Spawned object position is NaN; retrying ({verify_attempt + 1}/{max_verify_attempts})...")
                                    try:
                                        self.client.simContinueForFrames(1)
                                    except:
                                        pass
                                    continue
                                else:
                                    print(f"✗ Spawned object position is NaN")
                                    try:
                                        all_objects = self.client.simListSceneObjects(".*")
                                        matching_objects = [obj for obj in all_objects if self.target_object_name.lower() in obj.lower()]
                                        if matching_objects:
                                            print(f"  Matching objects: {matching_objects[:5]}")
                                        print(f"  Total scene objects: {len(all_objects)}")
                                    except:
                                        pass
                                    return False
                            
                            return True
                        else:
                            if verify_attempt < max_verify_attempts - 1:
                                safe_log(f"⚠ Object pose not ready; retrying ({verify_attempt + 1}/{max_verify_attempts})...", scene_id=self.scene_id)
                                try:
                                    self.client.simContinueForFrames(1)
                                except:
                                    pass
                                continue
                            else:
                                safe_log(f"⚠ Failed to verify spawned target object pose", scene_id=self.scene_id)
                                return False
                    except Exception as e:
                        if verify_attempt < max_verify_attempts - 1:
                            safe_log(f"⚠ Object pose verification failed ({verify_attempt + 1}/{max_verify_attempts}): {e}", scene_id=self.scene_id)
                            continue
                        else:
                            safe_log(f"✗ Target object spawn failed: {e}", scene_id=self.scene_id)
                            return False
                
                print(f"⚠ Failed to verify target object pose")
                return False
            else:
                print(f"✗ simSpawnObject failed")
                return False
                
        except Exception as e:
            print(f"✗ Operation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    

    def teleport_object_to_start(self, x, y, z):
        import time
        import numpy as np
        import airsim

        try:
            if not self.spawn_target_object(x, y, z):
                return False

            pose = self.client.simGetObjectPose(self.target_object_name)
            if pose is not None:
                p = pose.position
                cur = np.array([p.x_val, p.y_val, p.z_val], dtype=np.float32)
                err = float(np.linalg.norm(cur - np.array([float(x), float(y), float(z)], dtype=np.float32)))
            else:
                err = float("inf")

            if err > 1.0:
                ok, last, err2 = self._set_object_pose_paused(
                    self.target_object_name, x, y, z,
                    quat=airsim.to_quaternion(0, 0, 0),
                    retries=3,
                    tol=1.0
                )
                if not ok:
                    print(
                        f"✗ Target teleport failed: target=({float(x):.2f},{float(y):.2f},{float(z):.2f}), "
                        f"actual=({last[0]:.2f},{last[1]:.2f},{last[2]:.2f}), error={err2:.2f}m"
                    )
                    return False

            self._step_if_needed(1)
            return True

        except Exception as e:
            print(f"✗ Operation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_object_position(self):
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                pose = self.client.simGetObjectPose(self.target_object_name)
                if pose is None:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
                pos = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
                if np.any(np.isnan(pos)):
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
                return pos
            except (TimeoutError, Exception) as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
        
        return None
    
    def get_object_pose(self):
        try:
            pose = self.client.simGetObjectPose(self.target_object_name)
            if pose is None:
                return None, None
            position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
            if np.any(np.isnan(position)):
                return None, None
            orientation = np.array([pose.orientation.w_val, pose.orientation.x_val, 
                                    pose.orientation.y_val, pose.orientation.z_val])
            return position, orientation
        except Exception as e:
            return None, None
    
    def get_uav_state(self):
        state = self.client.getMultirotorState(vehicle_name=self.uav_vehicle_name)
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.uav_vehicle_name)
        has_collided = collision_info.has_collided if collision_info else False
        collision_time_stamp = None
        if collision_info and has_collided:
            try:
                if hasattr(collision_info, 'time_stamp'):
                    collision_time_stamp = collision_info.time_stamp
                elif hasattr(collision_info, 'time_stamp_ns'):
                    collision_time_stamp = collision_info.time_stamp_ns
            except:
                pass
        
        return {
            'position': np.array([pos.x_val, pos.y_val, pos.z_val]),
            'orientation': np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val]),
            'has_collided': has_collided,
            'collision_time_stamp': collision_time_stamp
        }
    
    def get_camera_images(self):
        max_retries = 5
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                responses_rgb = self.client.simGetImages([
                    airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)
                ], vehicle_name=self.uav_vehicle_name)
                
                rgb_response = responses_rgb[0]
                rgb_img = None
                if rgb_response.image_data_uint8:
                    rgb_img = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
                    rgb_img = rgb_img.reshape(rgb_response.height, rgb_response.width, 3)
                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                
                depth_img = None
                return rgb_img, depth_img
            except (TimeoutError, Exception) as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    safe_log(f"⚠ Image request failed ({attempt + 1}/{max_retries}): {error_msg}", scene_id=self.scene_id)
                    time.sleep(retry_delay)
                    continue
                else:
                    safe_log(f"✗ Image request failed after {max_retries} retries: {error_msg}", scene_id=self.scene_id)
                    import traceback
                    traceback.print_exc()
                    return None, None
        
        return None, None
    

    def move_target_object(self, target_pos):
        import airsim
        import numpy as np

        try:
            test_pose = self.client.simGetObjectPose(self.target_object_name)
            if test_pose is None:
                if self.target_asset_name is None:
                    self.target_asset_name = self.target_object_name
                if not self.spawn_target_object(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])):
                    safe_log(
                        f"⚠ Target object {self.target_object_name} is missing; respawning",
                        scene_id=self.scene_id
                    )
            else:
                test_pos = np.array([test_pose.position.x_val, test_pose.position.y_val, test_pose.position.z_val])
                if np.any(np.isnan(test_pos)):
                    if self.target_asset_name is None:
                        self.target_asset_name = self.target_object_name
                    if not self.spawn_target_object(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])):
                        safe_log(
                            f"⚠ Target object {self.target_object_name} pose is NaN; respawning",
                            scene_id=self.scene_id
                        )
        except Exception as e:
            if self.target_asset_name is None:
                self.target_asset_name = self.target_object_name
            try:
                self.spawn_target_object(float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
            except Exception as spawn_error:
                safe_log(
                    f"⚠ Failed to refresh target object {self.target_object_name}: {e}; respawn error: {spawn_error}",
                    scene_id=self.scene_id
                )

        pose_quat = airsim.to_quaternion(0, 0, 0)
        ok, last, err = self._set_object_pose_paused(
            self.target_object_name,
            float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
            quat=pose_quat,
            retries=2,
            tol=1.0
        )
        if not ok:
            if last is not None:
                safe_log(
                    f"⚠ Target move mismatch: target=({float(target_pos[0]):.2f},{float(target_pos[1]):.2f},{float(target_pos[2]):.2f}) "
                    f"actual=({last[0]:.2f},{last[1]:.2f},{last[2]:.2f}) err={err:.2f}m",
                    scene_id=self.scene_id
                )
            else:
                safe_log(
                    f"⚠ Target move mismatch: target=({float(target_pos[0]):.2f},{float(target_pos[1]):.2f},{float(target_pos[2]):.2f}) "
                    f"actual pose unavailable",
                    scene_id=self.scene_id
                )

    def cleanup_old_frames(self, dataset_dir):
        dataset_path = Path(dataset_dir)
        rgb_dir = dataset_path / "rgb"
        
        if rgb_dir.exists():
            for old_file in rgb_dir.glob("frame_*.png"):
                try:
                    old_file.unlink()
                except Exception:
                    pass
    
    def save_frame_data(self, frame_idx, rgb_img, depth_img, dataset_dir):
        dataset_path = Path(dataset_dir)
        
        rgb_dir = dataset_path / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        
        if rgb_img is not None:
            rgb_path = rgb_dir / f"frame_{frame_idx:05d}.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    
    def _prepare_target_object(self):
        if self._target_asset_name_explicitly_set:
            selected_uav_name = self.target_asset_name
        else:
            selected_uav_num = random.randint(1, 20)
            selected_uav_name = f"UAV{selected_uav_num}"
            self.target_asset_name = selected_uav_name
        
        import time as time_module
        unique_suffix = int(time_module.time() * 1000) % 100000
        random_suffix = random.randint(1000, 9999)
        unique_object_name = f"{selected_uav_name}_{unique_suffix}_{random_suffix}"
        
        self.target_object_name = unique_object_name
        
        return selected_uav_name
    
    def _prepare_dataset_directory(self, trajectory_name, dataset_base_dir, save_dataset):
        dataset_dir = None
        if save_dataset:
            dataset_path = Path(dataset_base_dir) / self.scene_id / trajectory_name
            dataset_path.mkdir(parents=True, exist_ok=True)
            dataset_dir = str(dataset_path)
            self.cleanup_old_frames(dataset_dir)
        return dataset_dir
    
    def _initialize_simulation(self, uav_traj, target_traj):
        self.connect()
        max_retries = 10
        base_retry_delay = 2
        scene_restart_threshold = 1
        last_error = None
        scene_restarted = False
        consecutive_timeouts = 0
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    try:
                        self.client.getMultirotorState(vehicle_name=self.uav_vehicle_name)
                        consecutive_timeouts = 0
                    except Exception as conn_e:
                        consecutive_timeouts += 1
                        safe_log(f"⚠ [{self.scene_id}] AirSim connection lost: {str(conn_e)}", scene_id=self.scene_id)
                        if "timeout" in str(conn_e).lower() or "timed out" in str(conn_e).lower():
                            sim_tool_to_use = self.pre_existing_sim_client_tool if self.pre_existing_sim_client_tool is not None else self.sim_client_tool
                            if not scene_restarted and sim_tool_to_use is not None:
                                safe_log(f"🔄 [{self.scene_id}] Restarting scene after timeout...", scene_id=self.scene_id)
                                try:
                                    try:
                                        if hasattr(sim_tool_to_use, '_closeConnection'):
                                            sim_tool_to_use._closeConnection()
                                    except:
                                        pass
                                    try:
                                        if hasattr(sim_tool_to_use, '_closeSocketConnection'):
                                            sim_tool_to_use._closeSocketConnection()
                                    except:
                                        pass
                                    
                                    time.sleep(3)
                                    
                                    main_scene_id = self.scene_id if isinstance(self.scene_id, str) else self.scene_id[0]
                                    machines_info = [{
                                        'MACHINE_IP': self.sim_server_host,
                                        'SOCKET_PORT': self.sim_server_port,
                                        'open_scenes': [main_scene_id] if isinstance(self.scene_id, str) else self.scene_id,
                                        'gpus': [self.gpu_id] if isinstance(self.gpu_id, (int, str)) else self.gpu_id
                                    }]
                                    if os.environ.get("DAGGER_MULTI_WORKER") != "1":
                                        try:
                                            tmp_client = msgpackrpc.Client(
                                                msgpackrpc.Address(self.sim_server_host, self.sim_server_port),
                                                timeout=30,
                                            )
                                            try:
                                                tmp_client.call('close_scenes', self.sim_server_host)
                                            except Exception:
                                                pass
                                            try:
                                                tmp_client.close()
                                            except Exception:
                                                pass
                                        except Exception:
                                            pass
                                    new_sim_client_tool = AirVLNSimulatorClientTool(machines_info)
                                    new_sim_client_tool.run_call()
                                    
                                    if self.pre_existing_sim_client_tool is not None:
                                        self.pre_existing_sim_client_tool = new_sim_client_tool
                                    self.sim_client_tool = new_sim_client_tool
                                    
                                    scene_index = self.scene_index if isinstance(self.scene_id, str) else 0
                                    if len(new_sim_client_tool.airsim_clients) > 0 and len(new_sim_client_tool.airsim_clients[0]) > scene_index:
                                        self.client = new_sim_client_tool.airsim_clients[0][scene_index]
                                    else:
                                        self.client = new_sim_client_tool.airsim_clients[0][0]
                                    
                                    if self.pre_existing_client is not None:
                                        self.pre_existing_client = self.client
                                    
                                    self._connected_scene_id = main_scene_id
                                    scene_restarted = True
                                    safe_log(f"✓ [{self.scene_id}] Scene restarted successfully", scene_id=self.scene_id)
                                    time.sleep(5)
                                    consecutive_timeouts = 0
                                except Exception as restart_e:
                                    safe_log(f"⚠ [{self.scene_id}] Scene restart failed: {str(restart_e)}", scene_id=self.scene_id)
                                    scene_restarted = True
                        else:
                            try:
                                self.connect(reuse_connection=False)
                            except:
                                pass
                
                sim_tool_to_use = self.pre_existing_sim_client_tool if self.pre_existing_sim_client_tool is not None else self.sim_client_tool
                if attempt >= scene_restart_threshold and not scene_restarted and sim_tool_to_use is not None:
                    safe_log(f"🔄 [{self.scene_id}] Restarting scene before enableApiControl (attempt {attempt})...", scene_id=self.scene_id)
                    try:
                        try:
                            if hasattr(sim_tool_to_use, '_closeConnection'):
                                sim_tool_to_use._closeConnection()
                        except:
                            pass
                        try:
                            if hasattr(sim_tool_to_use, '_closeSocketConnection'):
                                sim_tool_to_use._closeSocketConnection()
                        except:
                            pass
                        
                        time.sleep(3)
                        
                        main_scene_id = self.scene_id if isinstance(self.scene_id, str) else self.scene_id[0]
                        machines_info = [{
                            'MACHINE_IP': self.sim_server_host,
                            'SOCKET_PORT': self.sim_server_port,
                            'open_scenes': [main_scene_id] if isinstance(self.scene_id, str) else self.scene_id,
                            'gpus': [self.gpu_id] if isinstance(self.gpu_id, (int, str)) else self.gpu_id
                        }]
                        new_sim_client_tool = AirVLNSimulatorClientTool(machines_info)
                        new_sim_client_tool.run_call()
                        
                        if self.pre_existing_sim_client_tool is not None:
                            self.pre_existing_sim_client_tool = new_sim_client_tool
                        self.sim_client_tool = new_sim_client_tool
                        
                        scene_index = self.scene_index if isinstance(self.scene_id, str) else 0
                        if len(new_sim_client_tool.airsim_clients) > 0 and len(new_sim_client_tool.airsim_clients[0]) > scene_index:
                            self.client = new_sim_client_tool.airsim_clients[0][scene_index]
                        else:
                            self.client = new_sim_client_tool.airsim_clients[0][0]
                        
                        if self.pre_existing_client is not None:
                            self.pre_existing_client = self.client
                        
                        self._connected_scene_id = main_scene_id
                        scene_restarted = True
                        safe_log(f"✓ [{self.scene_id}] Scene restarted successfully", scene_id=self.scene_id)
                        time.sleep(5)
                    except Exception as restart_e:
                        safe_log(f"⚠ [{self.scene_id}] Scene restart failed: {str(restart_e)}", scene_id=self.scene_id)
                        scene_restarted = True
                
                self.client.enableApiControl(True, vehicle_name=self.uav_vehicle_name)
                if attempt > 0:
                    if scene_restarted:
                        safe_log(f"✓ [{self.scene_id}] enableApiControl succeeded on attempt {attempt+1}", scene_id=self.scene_id)
                    else:
                        safe_log(f"✓ [{self.scene_id}] enableApiControl succeeded on attempt {attempt+1}", scene_id=self.scene_id)
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    error_msg = str(e)
                    is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                    
                    if is_timeout:
                        consecutive_timeouts += 1
                        sim_tool_to_use = self.pre_existing_sim_client_tool if self.pre_existing_sim_client_tool is not None else self.sim_client_tool
                        if consecutive_timeouts >= 1 and not scene_restarted and sim_tool_to_use is not None:
                            safe_log(f"🔄 [{self.scene_id}] Restarting scene after {consecutive_timeouts} consecutive timeouts...", scene_id=self.scene_id)
                            try:
                                try:
                                    if hasattr(sim_tool_to_use, '_closeConnection'):
                                        sim_tool_to_use._closeConnection()
                                except:
                                    pass
                                try:
                                    if hasattr(sim_tool_to_use, '_closeSocketConnection'):
                                        sim_tool_to_use._closeSocketConnection()
                                except:
                                    pass
                                
                                time.sleep(3)
                                
                                main_scene_id = self.scene_id if isinstance(self.scene_id, str) else self.scene_id[0]
                                machines_info = [{
                                    'MACHINE_IP': self.sim_server_host,
                                    'SOCKET_PORT': self.sim_server_port,
                                    'open_scenes': [main_scene_id] if isinstance(self.scene_id, str) else self.scene_id,
                                    'gpus': [self.gpu_id] if isinstance(self.gpu_id, (int, str)) else self.gpu_id
                                }]
                                new_sim_client_tool = AirVLNSimulatorClientTool(machines_info)
                                new_sim_client_tool.run_call()
                                
                                if self.pre_existing_sim_client_tool is not None:
                                    self.pre_existing_sim_client_tool = new_sim_client_tool
                                self.sim_client_tool = new_sim_client_tool
                                
                                scene_index = self.scene_index if isinstance(self.scene_id, str) else 0
                                if len(new_sim_client_tool.airsim_clients) > 0 and len(new_sim_client_tool.airsim_clients[0]) > scene_index:
                                    self.client = new_sim_client_tool.airsim_clients[0][scene_index]
                                else:
                                    self.client = new_sim_client_tool.airsim_clients[0][0]
                                
                                if self.pre_existing_client is not None:
                                    self.pre_existing_client = self.client
                                
                                self._connected_scene_id = main_scene_id
                                scene_restarted = True
                                consecutive_timeouts = 0
                                safe_log(f"✓ [{self.scene_id}] Scene restarted successfully", scene_id=self.scene_id)
                                time.sleep(5)
                            except Exception as restart_e:
                                safe_log(f"⚠ [{self.scene_id}] Scene restart failed: {str(restart_e)}", scene_id=self.scene_id)
                                scene_restarted = True
                    else:
                        consecutive_timeouts = 0
                    
                    retry_delay = base_retry_delay * (attempt + 1)
                    if is_timeout:
                        safe_log(f"⚠ [{self.scene_id}] enableApiControl timeout; retrying in {retry_delay}s ({attempt+1}/{max_retries})", scene_id=self.scene_id)
                        time.sleep(retry_delay)
                        continue
                    else:
                        safe_log(f"⚠ [{self.scene_id}] enableApiControl failed: {error_msg[:100]}; retrying in {retry_delay}s ({attempt+1}/{max_retries})", scene_id=self.scene_id)
                        time.sleep(retry_delay)
                        continue
                else:
                    error_msg = str(last_error)
                    safe_log(f"❌ [{self.scene_id}] enableApiControl failed after {max_retries} retries: {error_msg[:200]}", scene_id=self.scene_id)
                    raise
        self.client.armDisarm(True, vehicle_name=self.uav_vehicle_name)
        
        u0 = uav_traj[0]
        t0 = target_traj[0]
        
        self.teleport_object_to_start(t0[0], t0[1], t0[2])
        
        self.teleport_to_start(u0[0], u0[1], u0[2], 
                              target_x=t0[0], target_y=t0[1], target_z=t0[2])
        
        try:
            max_retries = 3
            pos_error = float('inf')
            cur_pos = None
            
            for retry in range(max_retries):
                uav_state = self.get_uav_state()
                cur_pos = uav_state['position']
                
                pos_error = np.linalg.norm(cur_pos - np.array(u0))
                delta_xy = cur_pos[:2] - u0[:2]
                pos_error_xy = np.linalg.norm(delta_xy)
                pos_error_z = abs(cur_pos[2] - u0[2])
                
                if pos_error <= 1.0:
                    break
                
                if retry < max_retries - 1:
                    print(
                        f"⚠ UAV initial position mismatch (attempt {retry + 1}/{max_retries}): "
                        f"target=({u0[0]:.2f}, {u0[1]:.2f}, {u0[2]:.2f}), "
                        f"actual=({cur_pos[0]:.2f}, {cur_pos[1]:.2f}, {cur_pos[2]:.2f}), "
                        f"error={pos_error:.2f}m (XY: {pos_error_xy:.2f}m, Z: {pos_error_z:.2f}m)"
                    )
                    self.teleport_to_start(
                        u0[0], u0[1], u0[2],
                        target_x=t0[0], target_y=t0[1], target_z=t0[2]
                    )
                    try:
                        self.client.simContinueForFrames(5)
                    except:
                        pass
                    time.sleep(0.1)
            
            if pos_error > 1.0 and cur_pos is not None:
                final_pos_error_xy = np.linalg.norm(cur_pos[:2] - u0[:2])
                final_pos_error_z = abs(cur_pos[2] - u0[2])
                print(
                    f"⚠ UAV initial position still mismatched after retries: "
                    f"target=({u0[0]:.2f}, {u0[1]:.2f}, {u0[2]:.2f}), "
                    f"actual=({cur_pos[0]:.2f}, {cur_pos[1]:.2f}, {cur_pos[2]:.2f}), "
                    f"error={pos_error:.2f}m (XY: {final_pos_error_xy:.2f}m, Z: {final_pos_error_z:.2f}m)"
                )
            
            dx = t0[0] - cur_pos[0]
            dy = t0[1] - cur_pos[1]
            yaw = np.arctan2(dy, dx)
            
            quat = airsim.to_quaternion(0, 0, yaw)
            
            self.client.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(cur_pos[0], cur_pos[1], cur_pos[2]), quat),
                ignore_collision=True,
                vehicle_name=self.uav_vehicle_name
            )
            
            try:
                self.client.simContinueForFrames(1)
            except:
                pass
        except Exception as e:
            print(f"⚠ Initialization warning: {e}")
            import traceback
            traceback.print_exc()
    
    def _reset_collision_state(self):
        self.reset_collision_info()
    
    def _ensure_uav_flying_state(self):
        try:
            uav_state = self.get_uav_state()
            cur_pos = uav_state['position']
            
            pos2_now = self.get_object_position()
            if pos2_now is not None:
                dx = pos2_now[0] - cur_pos[0]
                dy = pos2_now[1] - cur_pos[1]
                yaw = np.arctan2(dy, dx)
            else:
                orientation = uav_state['orientation']
                yaw = 0.0
            
            quat = airsim.to_quaternion(0, 0, yaw)
            
            self.client.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(cur_pos[0], cur_pos[1], cur_pos[2]), quat),
                ignore_collision=True,
                vehicle_name=self.uav_vehicle_name
            )
        except:
            pass
    
    def _quaternion_to_euler(self, quat_w, quat_x, quat_y, quat_z):
        rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w])
        euler = rotation.as_euler('xyz', degrees=False)
        return {
            "roll": float(euler[0]),
            "pitch": float(euler[1]),
            "yaw": float(euler[2])
        }
    
    def _world_to_body_frame(self, vector_world, quat_w, quat_x, quat_y, quat_z):
        rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w])
        vector_body = rotation.inv().apply(vector_world)
        vector_body[2] = -vector_body[2]
        return vector_body

    def _airsim_to_body_frame(self, vector_world, quat_w, quat_x, quat_y, quat_z):
        rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w])
        vector_body = rotation.inv().apply(vector_world)
        return vector_body
    
    def _append_trajectory_data(self, frame_idx, uav_state, cur1_pos, pos2_now,
                                merged_trajectory_data, next_target_pos_airsim=None):
        uav_pos = np.array([
            float(cur1_pos[0]),
            float(cur1_pos[1]),
            float(-cur1_pos[2])
        ])
        
        uav_quat = uav_state['orientation']
        uav_quat_w = float(uav_quat[0])
        uav_quat_x = float(uav_quat[1])
        uav_quat_y = float(uav_quat[2])
        uav_quat_z = float(uav_quat[3])
        
        uav_euler = self._quaternion_to_euler(uav_quat_w, uav_quat_x, uav_quat_y, uav_quat_z)
        
        if self._prev_frame_data is not None and 'frame_data' in self._prev_frame_data:
            prev_frame_data = self._prev_frame_data['frame_data']
            prev_pos = self._prev_frame_data['uav_position']
            prev_quat = self._prev_frame_data['uav_orientation_quaternion']
            prev_euler = self._prev_frame_data['uav_orientation_euler']
            
            position_diff_world = uav_pos - np.array([prev_pos['x'], prev_pos['y'], prev_pos['z']])
            
            velocity_body = self._world_to_body_frame(
                position_diff_world,
                prev_quat['w'], prev_quat['x'], prev_quat['y'], prev_quat['z']
            )
            
            prev_yaw = prev_euler['yaw']
            current_yaw = uav_euler['yaw']
            yaw_diff = current_yaw - prev_yaw
            
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            
            yaw_rate_deg = np.degrees(yaw_diff)
            
            prev_frame_data["velocity_in_body_frame"] = {
                "x": float(velocity_body[0]),
                "y": float(velocity_body[1]),
                "z": float(velocity_body[2])
            }
            prev_frame_data["yaw_rate"] = float(yaw_rate_deg)
        
        frame_data = {
            "frame_idx": frame_idx,
            "uav_position": {
                "x": uav_pos[0],
                "y": uav_pos[1],
                "z": uav_pos[2]
            },
            "uav_orientation_quaternion": {
                "w": uav_quat_w,
                "x": uav_quat_x,
                "y": uav_quat_y,
                "z": uav_quat_z
            },
            "uav_orientation_euler": uav_euler
        }
        
        if next_target_pos_airsim is not None:
            next_target_pos_world = np.array([
                float(next_target_pos_airsim[0]),
                float(next_target_pos_airsim[1]),
                float(-next_target_pos_airsim[2])
            ])
            
            relative_position = next_target_pos_world - uav_pos
            
            relative_position_body = self._world_to_body_frame(
                relative_position,
                uav_quat_w, uav_quat_x, uav_quat_y, uav_quat_z
            )
            
            frame_data["target_position"] = {
                "x": next_target_pos_world[0],
                "y": next_target_pos_world[1],
                "z": next_target_pos_world[2]
            }
            frame_data["relative_position"] = {
                "x": float(relative_position[0]),
                "y": float(relative_position[1]),
                "z": float(relative_position[2])
            }
            frame_data["target_position_in_body_frame"] = {
                "x": float(relative_position_body[0]),
                "y": float(relative_position_body[1]),
                "z": float(relative_position_body[2])
            }
            distance = np.linalg.norm(relative_position)
            frame_data["distance"] = float(distance)
        else:
            frame_data["target_position"] = None
            frame_data["relative_position"] = None
            frame_data["target_position_in_body_frame"] = None
            frame_data["distance"] = None
        
        frame_data["velocity_in_body_frame"] = None
        frame_data["yaw_rate"] = None
        
        self._prev_frame_data = {
            "frame_data": frame_data,
            "uav_position": frame_data["uav_position"].copy(),
            "uav_orientation_quaternion": frame_data["uav_orientation_quaternion"].copy(),
            "uav_orientation_euler": frame_data["uav_orientation_euler"].copy()
        }
        
        merged_trajectory_data.append(frame_data)
    
    

    def _wrap_angle_rad(self, angle):
        """Wrap an angle in radians to [-pi, pi]."""
        return float(np.arctan2(np.sin(float(angle)), np.cos(float(angle))))

    def _target_facing_yaw(self, u_pos, t_pos, fallback_yaw=None):
        """
        Compute the horizontal yaw that points from UAV position to target position.

        This is used only to build expert yaw-rate labels from the planned expert
        trajectory. During frame execution for i > 0, the UAV orientation is updated
        by yaw_rate instead of being directly reset to face the target.
        """
        dx = float(t_pos[0]) - float(u_pos[0])
        dy = float(t_pos[1]) - float(u_pos[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return fallback_yaw
        return self._wrap_angle_rad(np.arctan2(dy, dx))

    def _expert_yaw_rate_for_step(self, uav_traj, target_traj, i):
        """
        Return the yaw-rate command, in degrees per frame, that should be applied
        when moving to frame i.

        Convention:
            frame 0: initialize yaw by facing the target, so no yaw_rate is applied.
            frame i>0: yaw_i = yaw_{i-1} + yaw_rate_{i-1}.

        Therefore the value returned here for step i is:
            yaw_rate_{i-1} = wrap(face_yaw_i - face_yaw_{i-1}) in degrees.
        """
        if i <= 0:
            return None
        if uav_traj is None or target_traj is None:
            return 0.0
        if i >= len(uav_traj) or i >= len(target_traj):
            return 0.0

        yaw_prev = self._target_facing_yaw(uav_traj[i - 1], target_traj[i - 1], fallback_yaw=None)
        yaw_cur = self._target_facing_yaw(uav_traj[i], target_traj[i], fallback_yaw=yaw_prev)
        if yaw_prev is None or yaw_cur is None:
            return 0.0

        yaw_delta = self._wrap_angle_rad(yaw_cur - yaw_prev)
        return float(np.degrees(yaw_delta))

    def _quat_from_yaw_rate(self, current_orientation, yaw_rate, step_idx=None):
        """
        Build a new UAV quaternion by integrating yaw_rate from the current yaw.

        yaw_rate is interpreted as degrees per frame, matching the saved
        `yaw_rate` field computed in `_append_trajectory_data`.
        """
        current_euler = self._quaternion_to_euler(
            float(current_orientation[0]),
            float(current_orientation[1]),
            float(current_orientation[2]),
            float(current_orientation[3]),
        )
        current_yaw = float(current_euler["yaw"])

        if yaw_rate is None:
            yaw_rate = 0.0
            if step_idx is not None:
                safe_log(
                    f"⚠ yaw_rate is None at step {step_idx}; keeping previous yaw instead of oracle-facing target.",
                    scene_id=self.scene_id,
                )

        yaw_next = self._wrap_angle_rad(current_yaw + np.deg2rad(float(yaw_rate)))
        return airsim.to_quaternion(0, 0, yaw_next)

    def _move_to_target_frame(self, u_target, t_target, i, num_steps, yaw_rate=None, jump_threshold=10.0):
        try:
            if i == 0:
                uav_state = None
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        uav_state = self.get_uav_state()
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            time.sleep(0.5)
                            continue
                        else:
                            raise RuntimeError(f"Failed to read UAV state: {e}")
                # Keep frame-0 orientation consistent with all later frames:
                # the UAV always faces the current target position in the horizontal plane.
                dx = float(t_target[0]) - float(u_target[0])
                dy = float(t_target[1]) - float(u_target[1])
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    current_orientation = uav_state['orientation']
                    quat = airsim.Quaternionr(
                        w_val=current_orientation[0],
                        x_val=current_orientation[1],
                        y_val=current_orientation[2],
                        z_val=current_orientation[3]
                    )
                else:
                    yaw = np.arctan2(dy, dx)
                    quat = airsim.to_quaternion(0, 0, yaw)
            else:
                uav_state = None
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        uav_state = self.get_uav_state()
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            time.sleep(0.5)
                            continue
                        else:
                            raise RuntimeError(f"Failed to read UAV state: {e}")
                cur_pos = uav_state['position']
                current_orientation = uav_state['orientation']
                
                jump_distance = np.linalg.norm(np.array([u_target[0], u_target[1], u_target[2]]) - cur_pos)
                if jump_distance > jump_threshold:
                    error_msg = f"Abnormal jump detected ({jump_distance:.2f}m > {jump_threshold}m) at step {i}; trajectory aborted"
                    safe_log(f"✗ {error_msg}", scene_id=self.scene_id)
                    raise RuntimeError(error_msg)
                else:
                    # Do NOT oracle-face the target after frame 0.
                    # For i > 0, the UAV heading is controlled by yaw_rate:
                    #     yaw_i = yaw_{i-1} + yaw_rate_{i-1}
                    # This makes the 4th action dimension a real control signal.
                    quat = self._quat_from_yaw_rate(
                        current_orientation=current_orientation,
                        yaw_rate=yaw_rate,
                        step_idx=i,
                    )

            max_position_retries = 3
            ok, verify_pos, pos_error, err_xy, err_z = self._set_vehicle_pose_paused(
                float(u_target[0]), float(u_target[1]), float(u_target[2]),
                quat,
                retries=max_position_retries,
                tol_xy=0.5,
                tol_z=0.5
            )
            if not ok:
                error_msg = (
                    f"Vehicle pose set failed: target=({u_target[0]:.2f}, {u_target[1]:.2f}, {u_target[2]:.2f}), "
                    f"actual=({verify_pos[0]:.2f}, {verify_pos[1]:.2f}, {verify_pos[2]:.2f}), "
                    f"error={pos_error:.2f}m (XY:{err_xy:.2f}m, Z:{err_z:.2f}m)"
                )
                raise RuntimeError(error_msg)

            
            self.reset_collision_info()
        except RuntimeError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "streamclosederror" not in error_msg and "connection" not in error_msg:
                safe_log(f"⚠ Failed to move target object: {e}", scene_id=self.scene_id)
        
        self.move_target_object(t_target)
        self._step_if_needed(1)
    
    def _process_frame(self, i, uav_traj, target_traj, trajectory_name, num_steps,
                      save_dataset, dataset_dir, merged_trajectory_data, pbar, target_trajectory_airsim=None):
        u_target = np.array([uav_traj[i][0], uav_traj[i][1], uav_traj[i][2]])
        t_target = np.array([target_traj[i][0], target_traj[i][1], target_traj[i][2]])
        
        yaw_rate_to_apply = self._expert_yaw_rate_for_step(uav_traj, target_traj, i)
        jump_threshold = getattr(self, '_jump_threshold', 10.0)
        self._move_to_target_frame(
            u_target, t_target, i, num_steps,
            yaw_rate=yaw_rate_to_apply,
            jump_threshold=jump_threshold,
        )
        
        uav_state = None
        max_retries = 3
        for retry in range(max_retries):
            try:
                uav_state = self.get_uav_state()
                break
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(0.5)
                    continue
                else:
                    safe_log(f"⚠ Failed to refresh UAV pose after step update: {e}", scene_id=self.scene_id)
                    uav_state = {
                        'position': u_target.copy(),
                        'orientation': np.array([1.0, 0.0, 0.0, 0.0]),
                        'has_collided': False
                    }
                    break
        
        cur1_pos = uav_state['position']
        
        pos2_now = None
        max_retries = 3
        for retry in range(max_retries):
            try:
                pos2_now = self.get_object_position()
                if pos2_now is not None:
                    break
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(0.5)
                    continue
        
        if pos2_now is None:
            pos2_now = t_target.copy()
        
        traj_num = trajectory_name.replace('trajectory_', '') if 'trajectory_' in trajectory_name else trajectory_name
        
        next_target_pos_airsim = None
        if i + 1 < len(target_traj):
            next_target_pos_airsim = np.array([
                target_traj[i + 1][0],
                target_traj[i + 1][1],
                target_traj[i + 1][2]
            ])
        
        if uav_state.get('has_collided', False):
            pbar.set_postfix_str("", refresh=True)
            tqdm.write("", file=sys.stderr)
            safe_log(f"⚠ Collision detected in trajectory {traj_num} at step {i}", scene_id=self.scene_id)
            raise RuntimeError(f"collision: trajectory {traj_num} step {i}")
        
        distance = np.linalg.norm(cur1_pos - pos2_now) if pos2_now is not None else 0
        
        try:
            if pos2_now is not None:
                pbar.set_postfix_str(
                    f"i={i}/{num_steps-1} "
                    f"D1=({cur1_pos[0]:.1f},{cur1_pos[1]:.1f},{-cur1_pos[2]:.1f}) "
                    f"T=({pos2_now[0]:.1f},{pos2_now[1]:.1f},{-pos2_now[2]:.1f}) "
                    f"dist={distance:.1f}m",
                    refresh=False
                )
            else:
                pbar.set_postfix_str(
                    f"i={i}/{num_steps-1} "
                    f"D1=({cur1_pos[0]:.1f},{cur1_pos[1]:.1f},{-cur1_pos[2]:.1f}) "
                    f"T=N/A",
                    refresh=False
                )
        except Exception:
            pass
        
        if save_dataset:
            rgb_img, depth_img = self.get_camera_images()
            self.save_frame_data(i, rgb_img, depth_img, dataset_dir)
        
        if target_trajectory_airsim is not None and pos2_now is not None:
            target_trajectory_airsim.append({
                "x": float(pos2_now[0]),
                "y": float(pos2_now[1]),
                "z": float(-pos2_now[2])
            })
        
        self._append_trajectory_data(i, uav_state, cur1_pos, pos2_now, 
                                     merged_trajectory_data, next_target_pos_airsim=next_target_pos_airsim)
    
    def _save_trajectory_files(self, dataset_dir, num_steps, selected_uav_name, 
                              merged_trajectory_data, save_dataset, target_trajectory_airsim=None,
                              planer_target_num_frames=None):
        if save_dataset:
            dataset_path = Path(dataset_dir)
            
            try:
                uav_traj_path = dataset_path / "uav_trajectory.json"
                temp_uav_path = dataset_path / "uav_trajectory.json.tmp"
                with open(temp_uav_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "num_frames": num_steps,
                        "target_asset_name": selected_uav_name,
                        "trajectory": merged_trajectory_data
                    }, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                temp_uav_path.replace(uav_traj_path)
            except Exception as e:
                safe_log(f"⚠ Failed to query UAV state: {e}", scene_id=self.scene_id)
                raise
            
            if target_trajectory_airsim is not None and len(target_trajectory_airsim) > 0:
                try:
                    target_traj_path = dataset_path / "target_trajectory.json"
                    temp_target_path = dataset_path / "target_trajectory.json.tmp"
                    target_num_frames = planer_target_num_frames if planer_target_num_frames is not None else len(target_trajectory_airsim)
                    
                    target_trajectory_save = []
                    for pos in target_trajectory_airsim:
                        if isinstance(pos, dict):
                            z_value = pos["z"]
                            if z_value < 0:
                                z_value = -z_value
                            target_trajectory_save.append({
                                "x": float(pos["x"]),
                                "y": float(pos["y"]),
                                "z": float(z_value)
                            })
                        else:
                            z_value = pos[2]
                            if z_value < 0:
                                z_value = -z_value
                            target_trajectory_save.append({
                                "x": float(pos[0]),
                                "y": float(pos[1]),
                                "z": float(z_value)
                            })
                    
                    with open(temp_target_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "num_frames": target_num_frames,
                            "target_trajectory_airsim": target_trajectory_save
                        }, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                    temp_target_path.replace(target_traj_path)
                except Exception as e:
                    safe_log(f"⚠ Processing warning: {e}", scene_id=self.scene_id)
    
    def _check_final_distance(self, trajectory_name, uav_traj, target_traj):
        traj_num = trajectory_name.replace('trajectory_', '') if 'trajectory_' in trajectory_name else trajectory_name
        
        num_steps = min(len(uav_traj), len(target_traj))
        if num_steps == 0:
            return
        
        pos1_final_arr = np.array([
            uav_traj[num_steps - 1][0],
            uav_traj[num_steps - 1][1],
            uav_traj[num_steps - 1][2]
        ])
        
        try:
            if hasattr(self, '_last_uav_position') and self._last_uav_position is not None:
                pos1_final_arr = self._last_uav_position
            else:
                try:
                    uav_state_final = self.get_uav_state()
                    pos1_final_arr = uav_state_final['position']
                    self._last_uav_position = pos1_final_arr
                except Exception:
                    pass
        except Exception:
            pass
        
        pos2_final_arr = np.array([
            target_traj[num_steps - 1][0],
            target_traj[num_steps - 1][1],
            target_traj[num_steps - 1][2]
        ])
        
        final_distance = np.linalg.norm(pos1_final_arr - pos2_final_arr)
        if final_distance >= 10.0:
            if hasattr(self, '_current_pbar') and self._current_pbar is not None:
                self._current_pbar.set_postfix_str("", refresh=True)
            tqdm.write("", file=sys.stderr)
            safe_log(f"⚠ Final UAV-target distance in trajectory {traj_num} is >= 10m ({final_distance:.2f}m)", scene_id=self.scene_id)
            raise RuntimeError(f"distance>=10m: trajectory {traj_num} final_distance={final_distance:.2f}m")
    
    def _cleanup_after_execution(self, skip_hover):
        if not skip_hover:
            try:
                print("\nHovering... (press Ctrl+C to stop)")
                print("  Current distance:")
                while True:
                    uav_state = self.get_uav_state()
                    pos1_arr = uav_state['position']
                    pos2_arr = self.get_object_position()
                    
                    if pos2_arr is not None:
                        dx = pos2_arr[0] - pos1_arr[0]
                        dy = pos2_arr[1] - pos1_arr[1]
                        yaw = np.arctan2(dy, dx)
                        current_distance = np.linalg.norm(pos1_arr - pos2_arr)
                        print(f"    {current_distance:.2f} m", end='\r')
                    else:
                        orientation = uav_state['orientation']
                        yaw = 0.0
                    
                    quat = airsim.to_quaternion(0, 0, yaw)
                    
                    self.client.simSetVehiclePose(
                        airsim.Pose(airsim.Vector3r(pos1_arr[0], pos1_arr[1], pos1_arr[2]), quat),
                        ignore_collision=True,
                        vehicle_name=self.uav_vehicle_name
                    )
            except KeyboardInterrupt:
                print("\n\nscene restart...")
            
            print("\n✓ Cleanup finished; AirSim connection closed")
        
        if self.client is not None:
            try:
                self.client.simDestroyObject(self.target_object_name)
            except Exception as e:
                print(f"⚠ Cleanup warning: {e}")
                try:
                    existing_objects = self.client.simListSceneObjects(self.target_object_name + ".*")
                    if existing_objects:
                        for obj_name in existing_objects:
                            try:
                                self.client.simDestroyObject(obj_name)
                                print(f"  Destroyed object: {obj_name}")
                            except:
                                pass
                except:
                    pass
    
    def execute_trajectory(self, trajectory_file, dataset_base_dir="/mnt/Data20T/ysq/OurVLN/Dataset", save_dataset=True, skip_hover=False, trajectory_index=None, total_trajectories=None, max_retries=5, jump_threshold=10.0):
        if not os.path.exists(trajectory_file):
            print(f"Starting trajectory execution: {trajectory_file}")
            return
        
        uav_traj, target_traj = self.load_trajectory(trajectory_file)
        
        if save_dataset:
            trajectory_name = Path(trajectory_file).stem
            if trajectory_name.endswith('_uav'):
                trajectory_name = trajectory_name[:-4]
            elif trajectory_name.endswith('_target'):
                trajectory_name = trajectory_name[:-7]
            
            dataset_path = Path(dataset_base_dir) / self.scene_id / trajectory_name
            rgb_dir = dataset_path / "rgb"
            uav_json_file = dataset_path / "uav_trajectory.json"
            
            num_steps = min(len(uav_traj), len(target_traj))
            if rgb_dir.exists() and uav_json_file.exists():
                existing_frames = []
                for frame_file in rgb_dir.glob("frame_*.png"):
                    try:
                        frame_num = int(frame_file.stem.split('_')[1])
                        existing_frames.append(frame_num)
                    except:
                        continue
                
                if len(existing_frames) >= num_steps:
                    expected_frames = set(range(num_steps))
                    saved_frames = set(existing_frames)
                    if expected_frames.issubset(saved_frames):
                        safe_log(f"⏭ [{self.scene_id}] Skipping trajectory {trajectory_name}; dataset already complete", scene_id=self.scene_id)
                        return
                    else:
                        safe_log(f"🔄 [{self.scene_id}] Resuming trajectory {trajectory_name} from existing frames ({len(existing_frames)}/{num_steps})", scene_id=self.scene_id)
                        try:
                            depth_dir = dataset_path / "depth"
                            for frame_file in rgb_dir.glob("frame_*.png"):
                                try:
                                    frame_file.unlink()
                                except:
                                    pass
                            if depth_dir.exists():
                                for frame_file in depth_dir.glob("frame_*.png"):
                                    try:
                                        frame_file.unlink()
                                    except:
                                        pass
                            for json_file in ['uav_trajectory.json', 'target_trajectory.json', 'jammer_trajectories.json', 'instruction.json']:
                                json_path = dataset_path / json_file
                                if json_path.exists():
                                    try:
                                        json_path.unlink()
                                    except:
                                        pass
                        except Exception as e:
                            safe_log(f"⚠ [{self.scene_id}] Dataset cleanup warning: {e}", scene_id=self.scene_id)
                elif len(existing_frames) > 0:
                    safe_log(f"🔄 [{self.scene_id}] Resuming trajectory {trajectory_name} from existing frames ({len(existing_frames)}/{num_steps})", scene_id=self.scene_id)
                    try:
                        depth_dir = dataset_path / "depth"
                        for frame_file in rgb_dir.glob("frame_*.png"):
                            try:
                                frame_file.unlink()
                            except:
                                pass
                        if depth_dir.exists():
                            for frame_file in depth_dir.glob("frame_*.png"):
                                try:
                                    frame_file.unlink()
                                except:
                                    pass
                        for json_file in ['uav_trajectory.json', 'target_trajectory.json', 'jammer_trajectories.json', 'instruction.json']:
                            json_path = dataset_path / json_file
                            if json_path.exists():
                                try:
                                    json_path.unlink()
                                except:
                                    pass
                    except Exception as e:
                        safe_log(f"⚠ [{self.scene_id}] Dataset cleanup warning: {e}", scene_id=self.scene_id)
        
        self._abnormal_jumps = []
        
        retry_count = 0
        while retry_count <= max_retries:
            try:
                return self._execute_trajectory_internal(trajectory_file, dataset_base_dir, save_dataset, skip_hover, trajectory_index, total_trajectories, uav_traj, target_traj, jump_threshold=jump_threshold)
            except RuntimeError as e:
                error_msg = str(e)
                if "collision" in error_msg.lower() or "distance>=10m" in error_msg.lower() or "abnormal jump" in error_msg.lower():
                    retry_count += 1
                    if retry_count <= max_retries:
                        safe_log(f"🔄 Retry {retry_count}: {error_msg}", scene_id=self.scene_id)
                        try:
                            self._cleanup_after_execution(skip_hover=True)
                        except:
                            pass
                        try:
                            trajectory_name = Path(trajectory_file).stem
                            if trajectory_name.endswith('_uav'):
                                trajectory_name = trajectory_name[:-4]
                            elif trajectory_name.endswith('_target'):
                                trajectory_name = trajectory_name[:-7]
                            dataset_path = Path(dataset_base_dir) / self.scene_id / trajectory_name
                            for json_file in ['uav_trajectory.json', 'target_trajectory.json', 'jammer_trajectories.json']:
                                json_path = dataset_path / json_file
                                if json_path.exists():
                                    json_path.unlink()
                        except:
                            pass
                        import time
                        time.sleep(0.5)
                        continue
                    else:
                        safe_log(f"✗ Execution failed after {max_retries} retries; giving up", scene_id=self.scene_id)
                        return
                else:
                    raise
            except Exception as e:
                raise
    
    def _execute_trajectory_internal(self, trajectory_file, dataset_base_dir, save_dataset, skip_hover, trajectory_index, total_trajectories, uav_traj, target_traj, jump_threshold=10.0):
        
        self._jump_threshold = jump_threshold
        
        self._abnormal_jumps = []
        
        selected_uav_name = self._prepare_target_object()
        
        trajectory_name = Path(trajectory_file).stem
        if trajectory_name.endswith('_uav'):
            trajectory_name = trajectory_name[:-4]
        elif trajectory_name.endswith('_target'):
            trajectory_name = trajectory_name[:-7]
        
        planer_target_num_frames = None
        planer_target_positions_airsim = None
        try:
            trajectory_path = Path(trajectory_file)
            if trajectory_path.name.endswith('_uav.json') or trajectory_path.name.endswith('_target.json'):
                base_name = trajectory_path.name.replace('_uav.json', '').replace('_target.json', '')
                planer_target_file = trajectory_path.parent / f"{base_name}_target.json"
            else:
                planer_target_file = trajectory_path
            
            if planer_target_file.exists():
                with open(planer_target_file, 'r', encoding='utf-8') as f:
                    planer_target_data = json.load(f)
                if 'target_trajectory' in planer_target_data and isinstance(planer_target_data['target_trajectory'], list):
                    planer_target_num_frames = len(planer_target_data['target_trajectory'])
                    planer_target_positions_airsim = []
                    for pos in planer_target_data['target_trajectory']:
                        if isinstance(pos, list) and len(pos) >= 3:
                            airsim_pos = {
                                "x": float(pos[0]),
                                "y": float(-pos[1]),
                                "z": float(-pos[2])
                            }
                            planer_target_positions_airsim.append(airsim_pos)
        except Exception as e:
            safe_log(f"⚠ Failed to parse planner target data: {e}; continuing without it", scene_id=self.scene_id)
        
        dataset_dir = self._prepare_dataset_directory(trajectory_name, dataset_base_dir, save_dataset)
        
        
        self._initialize_simulation(uav_traj, target_traj)
        
        self._reset_collision_state()
        
        num_steps = min(len(uav_traj), len(target_traj))
        merged_trajectory_data = []
        
        target_trajectory_airsim = []
        
        self._prev_frame_data = None
        
        try:
            if not self.client.isApiControlEnabled(vehicle_name=self.uav_vehicle_name):
                max_retries = 5
                retry_delay = 2
                for attempt in range(max_retries):
                    try:
                        self.client.enableApiControl(True, vehicle_name=self.uav_vehicle_name)
                        self.client.armDisarm(True, vehicle_name=self.uav_vehicle_name)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            error_msg = str(e)
                            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                                time.sleep(retry_delay)
                                continue
                        raise
        except:
            pass
        
        self._ensure_uav_flying_state()
        
        progress_position = getattr(self, '_progress_position', None)
        if trajectory_index is not None and total_trajectories is not None:
            desc = f"[{self.scene_id}] traj {trajectory_index}/{total_trajectories}"
        else:
            desc = f"[{self.scene_id}]"
        
        if progress_position is not None:
            pbar = tqdm(range(0, num_steps), 
                       desc=desc, 
                       unit="step", 
                       position=progress_position, 
                       leave=True, 
                       file=sys.stderr,
                       dynamic_ncols=True,
                       mininterval=0.1,
                       maxinterval=1.0)
            self._current_pbar = pbar
        else:
            pbar = tqdm(range(0, num_steps), 
                       desc=desc if trajectory_index is not None else "trajectory", 
                       unit="step",
                       file=sys.stderr,
                       dynamic_ncols=True,
                       mininterval=0.1)
            self._current_pbar = pbar
        
        for i in pbar:
            try:
                self._process_frame(i, uav_traj, target_traj, trajectory_name, num_steps,
                                  save_dataset, dataset_dir, merged_trajectory_data, pbar, target_trajectory_airsim)
                
            except RuntimeError as e:
                error_msg = str(e)
                if "collision" in error_msg.lower() or "distance>=10m" in error_msg.lower() or "abnormal jump" in error_msg.lower():
                    safe_log(f"✗ Step {i} failed: {e}", scene_id=self.scene_id)
                    try:
                        pbar.close()
                    except:
                        pass
                    raise
                else:
                    safe_log(f"✗ Step {i} failed: {e}", scene_id=self.scene_id)
                    continue
            except Exception as e:
                safe_log(f"✗ Step {i} failed: {e}", scene_id=self.scene_id)
                import traceback
                import io
                traceback_str = io.StringIO()
                traceback.print_exc(file=traceback_str)
                safe_log(traceback_str.getvalue(), scene_id=self.scene_id)
                continue
        
        try:
            if pbar is not None:
                pbar.refresh()
                pbar.close()
        except:
            pass
        finally:
            if hasattr(self, '_current_pbar'):
                self._current_pbar = None
        
        try:
            if num_steps > 0:
                self._last_uav_position = np.array([
                    uav_traj[num_steps - 1][0],
                    uav_traj[num_steps - 1][1],
                    uav_traj[num_steps - 1][2]
                ])
                try:
                    uav_state = self.get_uav_state()
                    self._last_uav_position = uav_state['position']
                except Exception:
                    pass
        except Exception:
            pass
        
        try:
            target_positions_to_save = planer_target_positions_airsim if planer_target_positions_airsim is not None else target_trajectory_airsim
            self._save_trajectory_files(dataset_dir, num_steps, selected_uav_name, 
                                       merged_trajectory_data, save_dataset, target_positions_to_save,
                                       planer_target_num_frames=planer_target_num_frames)
        except Exception as e:
            safe_log(f"⚠ Processing warning: {e}", scene_id=self.scene_id)
            import traceback
            traceback.print_exc()
        
        self._check_final_distance(trajectory_name, uav_traj, target_traj)
        
        try:
            self._cleanup_after_execution(skip_hover)
        except Exception as e:
            safe_log(f"⚠ Processing warning: {e}", scene_id=self.scene_id)




BaseTrajectoryExecutor = TrajectoryExecutor


class TrajectoryExecutor(BaseTrajectoryExecutor):
    def __init__(self,
                 scene_id="env_400",
                 sim_server_host="127.0.0.1",
                 sim_server_port=30000,
                 gpu_id=0,
                 scene_index=1,
                 uav_vehicle_name="Drone_1",
                 target_object_name="UAV1",
                 target_asset_name=None,
                 target_object_scale=(1.0, 1.0, 1.0),
                 camera_name="0",
                 auto_start_scene=True,
                 pre_existing_client=None,
                 pre_existing_sim_client_tool=None,
                 deterministic_step_mode=True,
                 jammer_enabled=True,
                 jammer_object_name="JammerUAV",
                 jammer_asset_name=None,
                 jammer_object_scale=(1.0, 1.0, 1.0)):
        super().__init__(
            scene_id=scene_id,
            sim_server_host=sim_server_host,
            sim_server_port=sim_server_port,
            gpu_id=gpu_id,
            scene_index=scene_index,
            uav_vehicle_name=uav_vehicle_name,
            target_object_name=target_object_name,
            target_asset_name=target_asset_name,
            target_object_scale=target_object_scale,
            camera_name=camera_name,
            auto_start_scene=auto_start_scene,
            pre_existing_client=pre_existing_client,
            pre_existing_sim_client_tool=pre_existing_sim_client_tool,
            deterministic_step_mode=deterministic_step_mode,
        )
        self.jammer_enabled = bool(jammer_enabled)
        self.jammer_object_name = jammer_object_name
        self.jammer_asset_name = jammer_asset_name if jammer_asset_name is not None else jammer_object_name
        self._jammer_asset_name_explicitly_set = (jammer_asset_name is not None)
        self.jammer_object_scale = jammer_object_scale
        self._selected_target_asset_name = None
        self._selected_jammer_asset_name = None
        self._all_jammer_trajectories_airsim = None
        self._primary_jammer_id = None
        self._jammer_object_names_by_id = {}
        self._jammer_asset_names_by_id = {}

    def _random_uav_asset_name(self):
        return f"UAV{random.randint(1, 20)}"

    def _available_uav_asset_names(self):
        return [f"UAV{i}" for i in range(1, 21)]

    def _sample_distinct_jammer_assets(self, jammer_ids, exclude_assets=None):
        jammer_ids = [str(x) for x in jammer_ids]
        exclude_assets = {str(x) for x in (exclude_assets or []) if x}
        explicit = bool(getattr(self, '_jammer_asset_name_explicitly_set', False))

        if explicit:
            fixed_asset = str(self.jammer_asset_name)
            return {did: fixed_asset for did in jammer_ids}

        pool = [name for name in self._available_uav_asset_names() if name not in exclude_assets]
        if len(pool) < len(jammer_ids):
            pool = self._available_uav_asset_names()

        shuffled = pool[:]
        random.shuffle(shuffled)
        selected = shuffled[:len(jammer_ids)]

        if len(selected) < len(jammer_ids):
            raise RuntimeError(
                f"Not enough UAV assets to assign distinct jammer assets: need {len(jammer_ids)}, have {len(pool)}"
            )

        return {did: asset for did, asset in zip(jammer_ids, selected)}

    def _prepare_named_object(self, object_name_attr, asset_name_attr, asset_explicit_attr):
        if getattr(self, asset_explicit_attr):
            selected_uav_name = getattr(self, asset_name_attr)
        else:
            selected_uav_name = self._random_uav_asset_name()
            setattr(self, asset_name_attr, selected_uav_name)

        import time as time_module
        unique_suffix = int(time_module.time() * 1000) % 100000
        random_suffix = random.randint(1000, 9999)
        unique_object_name = f"{selected_uav_name}_{unique_suffix}_{random_suffix}"
        setattr(self, object_name_attr, unique_object_name)
        return selected_uav_name

    def _prepare_target_object(self):
        selected_uav_name = self._prepare_named_object(
            object_name_attr="target_object_name",
            asset_name_attr="target_asset_name",
            asset_explicit_attr="_target_asset_name_explicitly_set",
        )
        self._selected_target_asset_name = selected_uav_name
        return selected_uav_name

    def _prepare_jammer_object(self):
        if not self.jammer_enabled:
            self._selected_jammer_asset_name = None
            return None
        selected_uav_name = self._prepare_named_object(
            object_name_attr="jammer_object_name",
            asset_name_attr="jammer_asset_name",
            asset_explicit_attr="_jammer_asset_name_explicitly_set",
        )
        self._selected_jammer_asset_name = selected_uav_name
        return selected_uav_name

    def _extract_trajectory_base_name(self, json_path):
        json_path = Path(json_path)
        name = json_path.name
        suffixes = [
            '_uav.json', '_target.json', '_jammer.json', '_interferer.json',
            '_disturb.json', '_decoy.json'
        ]
        for suffix in suffixes:
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return json_path.stem

    def _trajectory_to_airsim_array(self, traj_data, dataset_format=False):
        if traj_data is None:
            return None
        arr = []
        for pos in traj_data:
            if isinstance(pos, dict):
                if not all(k in pos for k in ("x", "y", "z")):
                    continue
                x = float(pos["x"])
                y = float(pos["y"])
                z = float(pos["z"])
            elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
                x = float(pos[0])
                y = float(pos[1])
                z = float(pos[2])
            else:
                continue

            if dataset_format:
                arr.append([x, y, -z])
            else:
                arr.append([x, -y, -z])

        if not arr:
            return None
        return np.asarray(arr, dtype=np.float32)

    def _load_optional_sidecar_trajectory(self, json_path, prefixes=("jammer", "interferer", "disturb", "decoy")):
        json_path = Path(json_path)
        base_name = self._extract_trajectory_base_name(json_path)
        candidates = []
        for prefix in prefixes:
            candidates.append(json_path.parent / f"{base_name}_{prefix}.json")
        for prefix in prefixes:
            candidates.append(json_path.parent / f"{prefix}_trajectory.json")

        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue

            for prefix in prefixes:
                key = f"{prefix}_trajectory"
                if key in data and isinstance(data[key], list):
                    traj = self._trajectory_to_airsim_array(data[key], dataset_format=False)
                    if traj is not None:
                        return traj, candidate

                key_airsim = f"{prefix}_trajectory_airsim"
                if key_airsim in data and isinstance(data[key_airsim], list):
                    traj = self._trajectory_to_airsim_array(data[key_airsim], dataset_format=True)
                    if traj is not None:
                        return traj, candidate

            if 'trajectory' in data and isinstance(data['trajectory'], list):
                for prefix in prefixes:
                    field = f"{prefix}_position"
                    traj_raw = []
                    for frame in data['trajectory']:
                        if isinstance(frame, dict) and field in frame and frame[field] is not None:
                            traj_raw.append(frame[field])
                    traj = self._trajectory_to_airsim_array(traj_raw, dataset_format=True)
                    if traj is not None:
                        return traj, candidate

        return None, None

    def _load_jammer_from_main_trajectory(self, json_path):
        """
        Fallback jammer trajectory loader.
        Supports all of the following frame-level formats:
          1) {"jammers": [{"id": 1, "position": [x, y, z]}, ...]}
          2) {"distractors": [{"id": 1, "position": [x, y, z]}, ...]}
          3) {"jammer_position": [x, y, z]}

        Priority:
          jammers/distractors list  >  single jammer_position field
        """
        json_path = Path(json_path)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return None

        frames = data.get("frames")
        if not isinstance(frames, list) or len(frames) == 0:
            return None

        traj_raw = []
        selected_jammer_id = None

        for frame in frames:
            if not isinstance(frame, dict):
                continue

            jammer_list = None
            for key in ("jammers", "distractors"):
                value = frame.get(key)
                if isinstance(value, list) and len(value) > 0:
                    jammer_list = value
                    break

            selected = None
            if jammer_list is not None:
                if selected_jammer_id is not None:
                    for item in jammer_list:
                        if isinstance(item, dict) and str(item.get("id")) == str(selected_jammer_id) and item.get("position") is not None:
                            selected = item
                            break

                if selected is None:
                    for item in jammer_list:
                        if isinstance(item, dict) and item.get("position") is not None:
                            selected = item
                            break

                if selected is not None:
                    if selected_jammer_id is None:
                        selected_jammer_id = selected.get("id")
                    pos = selected.get("position")
                    if pos is not None:
                        traj_raw.append(pos)
                    continue

            single_pos = frame.get("jammer_position")
            if single_pos is not None:
                traj_raw.append(single_pos)

        if not traj_raw:
            return None

        # Frames in the main planner/export trajectory use planner/world coordinates,
        # so they must be converted the same way as UAV/target trajectories: y -> -y, z -> -z.
        # Using dataset_format=True here would only flip z and mirror jammer positions on the y axis.
        return self._trajectory_to_airsim_array(traj_raw, dataset_format=False)

    def _load_all_jammers_from_main_trajectory(self, json_path):
        json_path = Path(json_path)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return {}

        frames = data.get("frames")
        if not isinstance(frames, list) or len(frames) == 0:
            return {}

        raw_by_id = {}
        for frame in frames:
            if not isinstance(frame, dict):
                continue

            jammer_list = None
            for key in ("jammers", "distractors"):
                value = frame.get(key)
                if isinstance(value, list):
                    jammer_list = value
                    break

            if not isinstance(jammer_list, list):
                continue

            for item in jammer_list:
                if not isinstance(item, dict):
                    continue
                did = item.get("id")
                pos = item.get("position")
                if did is None or pos is None:
                    continue
                key = str(did)
                if key not in raw_by_id:
                    raw_by_id[key] = []
                raw_by_id[key].append(pos)

        traj_by_id = {}
        for did, raw in raw_by_id.items():
            # Main-trajectory jammer positions come from planner/world coordinates,
            # not saved dataset coordinates, so convert with y -> -y and z -> -z.
            traj = self._trajectory_to_airsim_array(raw, dataset_format=False)
            if traj is not None and len(traj) > 0:
                traj_by_id[did] = traj
        return traj_by_id

    def load_trajectory_bundle(self, json_path):
        uav_traj, target_traj = super().load_trajectory(json_path)
        jammer_traj = None
        self._all_jammer_trajectories_airsim = None
        self._primary_jammer_id = None
        if self.jammer_enabled:
            jammer_traj, jammer_src = self._load_optional_sidecar_trajectory(json_path)
            if jammer_traj is not None:
                safe_log(f"✓ Loaded jammer sidecar trajectory: {Path(jammer_src).name}", scene_id=self.scene_id)
                self._all_jammer_trajectories_airsim = {"1": jammer_traj}
                self._primary_jammer_id = "1"
            else:
                self._all_jammer_trajectories_airsim = self._load_all_jammers_from_main_trajectory(json_path)
                if self._all_jammer_trajectories_airsim:
                    sorted_ids = sorted(
                        self._all_jammer_trajectories_airsim.keys(),
                        key=lambda x: int(x) if str(x).isdigit() else str(x),
                    )
                    self._primary_jammer_id = str(sorted_ids[0])
                    jammer_traj = self._all_jammer_trajectories_airsim[self._primary_jammer_id]
                else:
                    jammer_traj = self._load_jammer_from_main_trajectory(json_path)
        return uav_traj, target_traj, jammer_traj

    def _spawn_named_object(self, object_name, asset_name, object_scale, x, y, z):
        try:
            try:
                self.client.simDestroyObject(object_name)
                try:
                    self.client.simContinueForFrames(1)
                except Exception:
                    pass
            except Exception:
                try:
                    pattern = object_name + ".*"
                    existing_objects = self.client.simListSceneObjects(pattern)
                    for obj_name in existing_objects:
                        try:
                            self.client.simDestroyObject(obj_name)
                        except Exception:
                            pass
                    try:
                        self.client.simContinueForFrames(1)
                    except Exception:
                        pass
                except Exception:
                    pass

            pose = airsim.Pose(
                airsim.Vector3r(float(x), float(y), float(z)),
                airsim.to_quaternion(0, 0, 0)
            )
            scale_vector = airsim.Vector3r(float(object_scale[0]), float(object_scale[1]), float(object_scale[2]))
            success = self.client.simSpawnObject(
                object_name,
                asset_name,
                pose,
                scale_vector,
                physics_enabled=False,
                is_blueprint=False,
            )
            if not success:
                return False

            try:
                self.client.simContinueForFrames(1)
            except Exception:
                pass

            verify_pose = self.client.simGetObjectPose(object_name)
            return verify_pose is not None
        except Exception as e:
            safe_log(f"✗ object spawn failed {object_name}: {e}", scene_id=self.scene_id)
            return False

    def _set_named_object_pose_paused(self, object_name, x, y, z, quat=None, retries=3, tol=1.0):
        return self._set_object_pose_paused(object_name, x, y, z, quat=quat, retries=retries, tol=tol)

    def _teleport_named_object_to_start(self, object_name, asset_name, object_scale, x, y, z):
        try:
            if not self._spawn_named_object(object_name, asset_name, object_scale, x, y, z):
                return False
            pose = self.client.simGetObjectPose(object_name)
            if pose is not None:
                p = pose.position
                cur = np.array([p.x_val, p.y_val, p.z_val], dtype=np.float32)
                err = float(np.linalg.norm(cur - np.array([float(x), float(y), float(z)], dtype=np.float32)))
            else:
                err = float('inf')

            if err > 1.0:
                ok, last, err2 = self._set_named_object_pose_paused(
                    object_name, x, y, z,
                    quat=airsim.to_quaternion(0, 0, 0),
                    retries=3,
                    tol=1.0,
                )
                if not ok:
                    safe_log(
                        f"✗ object teleport failed {object_name}: target=({float(x):.2f},{float(y):.2f},{float(z):.2f}) err={err2:.2f}m",
                        scene_id=self.scene_id,
                    )
                    return False

            self._step_if_needed(1)
            return True
        except Exception as e:
            safe_log(f"✗ object teleport failed {object_name}: {e}", scene_id=self.scene_id)
            return False

    def teleport_jammer_to_start(self, x, y, z):
        if not self.jammer_enabled:
            return False
        return self._teleport_named_object_to_start(
            self.jammer_object_name,
            self.jammer_asset_name,
            self.jammer_object_scale,
            x, y, z,
        )

    def get_named_object_position(self, object_name):
        max_retries = 3
        retry_delay = 0.5
        for attempt in range(max_retries):
            try:
                pose = self.client.simGetObjectPose(object_name)
                if pose is None:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
                pos = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val], dtype=np.float32)
                if np.any(np.isnan(pos)):
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
                return pos
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
        return None

    def get_jammer_position(self):
        if not self.jammer_enabled:
            return None
        return self.get_named_object_position(self.jammer_object_name)

    def move_named_object(self, object_name, asset_name, object_scale, target_pos):
        try:
            test_pose = self.client.simGetObjectPose(object_name)
            if test_pose is None:
                if not self._spawn_named_object(object_name, asset_name, object_scale,
                                                float(target_pos[0]), float(target_pos[1]), float(target_pos[2])):
                    safe_log(f"⚠ failed to respawn {object_name}", scene_id=self.scene_id)
            else:
                test_pos = np.array([test_pose.position.x_val, test_pose.position.y_val, test_pose.position.z_val])
                if np.any(np.isnan(test_pos)):
                    if not self._spawn_named_object(object_name, asset_name, object_scale,
                                                    float(target_pos[0]), float(target_pos[1]), float(target_pos[2])):
                        safe_log(f"⚠ failed to respawn {object_name} from NaN pose", scene_id=self.scene_id)
        except Exception as e:
            try:
                self._spawn_named_object(object_name, asset_name, object_scale,
                                         float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
            except Exception as spawn_error:
                safe_log(f"⚠ failed to recover {object_name}: {e}; respawn error: {spawn_error}", scene_id=self.scene_id)

        ok, last, err = self._set_named_object_pose_paused(
            object_name,
            float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
            quat=airsim.to_quaternion(0, 0, 0),
            retries=2,
            tol=1.0,
        )
        if not ok:
            if last is not None:
                safe_log(
                    f"⚠ move object failed {object_name}: target=({float(target_pos[0]):.2f},{float(target_pos[1]):.2f},{float(target_pos[2]):.2f}) "
                    f"current=({last[0]:.2f},{last[1]:.2f},{last[2]:.2f}) err={err:.2f}m",
                    scene_id=self.scene_id,
                )
            else:
                safe_log(
                    f"⚠ move object failed {object_name}: target=({float(target_pos[0]):.2f},{float(target_pos[1]):.2f},{float(target_pos[2]):.2f})",
                    scene_id=self.scene_id,
                )

    def move_jammer_object(self, jammer_pos):
        if not self.jammer_enabled or jammer_pos is None:
            return
        self.move_named_object(
            self.jammer_object_name,
            self.jammer_asset_name,
            self.jammer_object_scale,
            jammer_pos,
        )

    def _prepare_all_jammer_objects(self, jammer_trajs_by_id):
        self._jammer_object_names_by_id = {}
        self._jammer_asset_names_by_id = {}
        if not self.jammer_enabled or not jammer_trajs_by_id:
            return None

        sorted_ids = sorted(
            [str(k) for k in jammer_trajs_by_id.keys()],
            key=lambda x: int(x) if str(x).isdigit() else str(x),
        )
        primary_id = self._primary_jammer_id if self._primary_jammer_id in jammer_trajs_by_id else str(sorted_ids[0])
        self._primary_jammer_id = primary_id

        exclude_assets = []
        if self._selected_target_asset_name:
            exclude_assets.append(self._selected_target_asset_name)

        asset_by_id = self._sample_distinct_jammer_assets(sorted_ids, exclude_assets=exclude_assets)
        primary_asset = asset_by_id.get(primary_id)
        if primary_asset is None:
            return None

        import time as time_module
        unique_suffix = int(time_module.time() * 1000) % 100000
        random_suffix = random.randint(1000, 9999)
        primary_object_name = f"{primary_asset}_{unique_suffix}_{random_suffix}"
        self.jammer_object_name = primary_object_name
        self.jammer_asset_name = primary_asset
        self._selected_jammer_asset_name = primary_asset
        self._jammer_object_names_by_id[primary_id] = primary_object_name
        self._jammer_asset_names_by_id[primary_id] = primary_asset

        for did in sorted_ids:
            did = str(did)
            if did == primary_id:
                continue
            asset_name = asset_by_id[did]
            unique_suffix = int(time_module.time() * 1000) % 100000
            random_suffix = random.randint(1000, 9999)
            object_name = f"{asset_name}_{unique_suffix}_{random_suffix}_j{did}"
            self._jammer_object_names_by_id[did] = object_name
            self._jammer_asset_names_by_id[did] = asset_name

        return dict(self._jammer_asset_names_by_id)

    def _initialize_simulation(self, uav_traj, target_traj, jammer_traj=None, jammer_trajs_by_id=None):
        super()._initialize_simulation(uav_traj, target_traj)
        if self.jammer_enabled and jammer_trajs_by_id:
            for did, traj in jammer_trajs_by_id.items():
                if traj is None or len(traj) == 0:
                    continue
                did = str(did)
                object_name = self._jammer_object_names_by_id.get(did, self.jammer_object_name)
                asset_name = self._jammer_asset_names_by_id.get(did, self.jammer_asset_name)
                j0 = traj[0]
                self._teleport_named_object_to_start(
                    object_name,
                    asset_name,
                    self.jammer_object_scale,
                    j0[0], j0[1], j0[2]
                )
        elif self.jammer_enabled and jammer_traj is not None and len(jammer_traj) > 0:
            j0 = jammer_traj[0]
            self.teleport_jammer_to_start(j0[0], j0[1], j0[2])

    def _move_to_target_frame(self, u_target, t_target, i, num_steps, yaw_rate=None, jump_threshold=10.0, j_target=None, j_targets_by_id=None):
        try:
            if i == 0:
                uav_state = None
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        uav_state = self.get_uav_state()
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            time.sleep(0.5)
                            continue
                        else:
                            raise RuntimeError(f"Failed to read UAV state: {e}")
                # Keep frame-0 orientation consistent with all later frames:
                # the UAV always faces the current target position in the horizontal plane.
                dx = float(t_target[0]) - float(u_target[0])
                dy = float(t_target[1]) - float(u_target[1])
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    current_orientation = uav_state['orientation']
                    quat = airsim.Quaternionr(
                        w_val=current_orientation[0],
                        x_val=current_orientation[1],
                        y_val=current_orientation[2],
                        z_val=current_orientation[3]
                    )
                else:
                    yaw = np.arctan2(dy, dx)
                    quat = airsim.to_quaternion(0, 0, yaw)
            else:
                uav_state = None
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        uav_state = self.get_uav_state()
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            time.sleep(0.5)
                            continue
                        else:
                            raise RuntimeError(f"Failed to read UAV state: {e}")
                cur_pos = uav_state['position']
                current_orientation = uav_state['orientation']

                jump_distance = np.linalg.norm(np.array([u_target[0], u_target[1], u_target[2]]) - cur_pos)
                if jump_distance > jump_threshold:
                    error_msg = f"Abnormal jump detected ({jump_distance:.2f}m > {jump_threshold}m) at step {i}; trajectory aborted"
                    safe_log(f"✗ {error_msg}", scene_id=self.scene_id)
                    raise RuntimeError(error_msg)
                else:
                    # Do NOT oracle-face the target after frame 0.
                    # For i > 0, the UAV heading is controlled by yaw_rate:
                    #     yaw_i = yaw_{i-1} + yaw_rate_{i-1}
                    # This makes the 4th action dimension a real control signal.
                    quat = self._quat_from_yaw_rate(
                        current_orientation=current_orientation,
                        yaw_rate=yaw_rate,
                        step_idx=i,
                    )

            max_position_retries = 3
            ok, verify_pos, pos_error, err_xy, err_z = self._set_vehicle_pose_paused(
                float(u_target[0]), float(u_target[1]), float(u_target[2]),
                quat,
                retries=max_position_retries,
                tol_xy=0.5,
                tol_z=0.5,
            )
            if not ok:
                error_msg = (
                    f"Vehicle pose set failed: target=({u_target[0]:.2f}, {u_target[1]:.2f}, {u_target[2]:.2f}), "
                    f"actual=({verify_pos[0]:.2f}, {verify_pos[1]:.2f}, {verify_pos[2]:.2f}), "
                    f"error={pos_error:.2f}m (XY:{err_xy:.2f}m, Z:{err_z:.2f}m)"
                )
                raise RuntimeError(error_msg)

            self.reset_collision_info()
        except RuntimeError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "streamclosederror" not in error_msg and "connection" not in error_msg:
                safe_log(f"⚠ Failed to move target object: {e}", scene_id=self.scene_id)

        self.move_target_object(t_target)
        if j_targets_by_id:
            for did, jpos in j_targets_by_id.items():
                did = str(did)
                object_name = self._jammer_object_names_by_id.get(did, self.jammer_object_name)
                asset_name = self._jammer_asset_names_by_id.get(did, self.jammer_asset_name)
                self.move_named_object(
                    object_name,
                    asset_name,
                    self.jammer_object_scale,
                    jpos,
                )
        elif j_target is not None:
            self.move_jammer_object(j_target)
        self._step_if_needed(1)

    def _append_trajectory_data(self, frame_idx, uav_state, cur1_pos, pos2_now,
                                merged_trajectory_data, next_target_pos_airsim=None):
        uav_pos = np.array([
            float(cur1_pos[0]),
            float(cur1_pos[1]),
            float(-cur1_pos[2])
        ])

        uav_quat = uav_state['orientation']
        uav_quat_w = float(uav_quat[0])
        uav_quat_x = float(uav_quat[1])
        uav_quat_y = float(uav_quat[2])
        uav_quat_z = float(uav_quat[3])

        uav_euler = self._quaternion_to_euler(uav_quat_w, uav_quat_x, uav_quat_y, uav_quat_z)

        if self._prev_frame_data is not None and 'frame_data' in self._prev_frame_data:
            prev_frame_data = self._prev_frame_data['frame_data']
            prev_pos = self._prev_frame_data['uav_position']
            prev_quat = self._prev_frame_data['uav_orientation_quaternion']
            prev_euler = self._prev_frame_data['uav_orientation_euler']

            position_diff_world = uav_pos - np.array([prev_pos['x'], prev_pos['y'], prev_pos['z']])
            velocity_body = self._world_to_body_frame(
                position_diff_world,
                prev_quat['w'], prev_quat['x'], prev_quat['y'], prev_quat['z']
            )

            prev_yaw = prev_euler['yaw']
            current_yaw = uav_euler['yaw']
            yaw_diff = current_yaw - prev_yaw
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            yaw_rate_deg = np.degrees(yaw_diff)

            prev_frame_data["velocity_in_body_frame"] = {
                "x": float(velocity_body[0]),
                "y": float(velocity_body[1]),
                "z": float(velocity_body[2])
            }
            prev_frame_data["yaw_rate"] = float(yaw_rate_deg)

        frame_data = {
            "frame_idx": frame_idx,
            "uav_position": {
                "x": uav_pos[0],
                "y": uav_pos[1],
                "z": uav_pos[2]
            },
            "uav_orientation_quaternion": {
                "w": uav_quat_w,
                "x": uav_quat_x,
                "y": uav_quat_y,
                "z": uav_quat_z
            },
            "uav_orientation_euler": uav_euler
        }

        if pos2_now is not None:
            target_pos_world = np.array([
                float(pos2_now[0]),
                float(pos2_now[1]),
                float(-pos2_now[2])
            ])
            relative_position = target_pos_world - uav_pos
            relative_position_body = self._world_to_body_frame(
                relative_position,
                uav_quat_w, uav_quat_x, uav_quat_y, uav_quat_z
            )
            frame_data["target_position"] = {
                "x": float(target_pos_world[0]),
                "y": float(target_pos_world[1]),
                "z": float(target_pos_world[2])
            }
            frame_data["relative_position"] = {
                "x": float(relative_position[0]),
                "y": float(relative_position[1]),
                "z": float(relative_position[2])
            }
            frame_data["target_position_in_body_frame"] = {
                "x": float(relative_position_body[0]),
                "y": float(relative_position_body[1]),
                "z": float(relative_position_body[2])
            }
            frame_data["distance"] = float(np.linalg.norm(relative_position))
        else:
            frame_data["target_position"] = None
            frame_data["relative_position"] = None
            frame_data["target_position_in_body_frame"] = None
            frame_data["distance"] = None

        if next_target_pos_airsim is not None:
            next_target_pos_world = np.array([
                float(next_target_pos_airsim[0]),
                float(next_target_pos_airsim[1]),
                float(-next_target_pos_airsim[2])
            ])
            frame_data["next_target_position"] = {
                "x": float(next_target_pos_world[0]),
                "y": float(next_target_pos_world[1]),
                "z": float(next_target_pos_world[2])
            }
        else:
            frame_data["next_target_position"] = None

        frame_data["velocity_in_body_frame"] = None
        frame_data["yaw_rate"] = None

        self._prev_frame_data = {
            "frame_data": frame_data,
            "uav_position": frame_data["uav_position"].copy(),
            "uav_orientation_quaternion": frame_data["uav_orientation_quaternion"].copy(),
            "uav_orientation_euler": frame_data["uav_orientation_euler"].copy()
        }
        merged_trajectory_data.append(frame_data)

    def _process_frame(self, i, uav_traj, target_traj, trajectory_name, num_steps,
                      save_dataset, dataset_dir, merged_trajectory_data, pbar,
                      target_trajectory_airsim=None, jammer_traj=None, jammer_trajectory_airsim=None,
                      jammer_trajs_by_id=None, jammer_trajectories_airsim_by_id=None):
        u_target = np.array([uav_traj[i][0], uav_traj[i][1], uav_traj[i][2]])
        t_target = np.array([target_traj[i][0], target_traj[i][1], target_traj[i][2]])
        j_target = None
        if jammer_traj is not None and i < len(jammer_traj):
            j_target = np.array([jammer_traj[i][0], jammer_traj[i][1], jammer_traj[i][2]])
        j_targets_by_id = None
        if jammer_trajs_by_id:
            j_targets_by_id = {}
            for did, traj in jammer_trajs_by_id.items():
                if traj is not None and i < len(traj):
                    j_targets_by_id[str(did)] = np.array([traj[i][0], traj[i][1], traj[i][2]])

        yaw_rate_to_apply = self._expert_yaw_rate_for_step(uav_traj, target_traj, i)
        jump_threshold = getattr(self, '_jump_threshold', 10.0)
        self._move_to_target_frame(
            u_target, t_target, i, num_steps,
            yaw_rate=yaw_rate_to_apply,
            jump_threshold=jump_threshold,
            j_target=j_target,
            j_targets_by_id=j_targets_by_id,
        )

        uav_state = None
        max_retries = 3
        for retry in range(max_retries):
            try:
                uav_state = self.get_uav_state()
                break
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(0.5)
                    continue
                else:
                    safe_log(f"⚠ Failed to refresh UAV pose after step update: {e}", scene_id=self.scene_id)
                    uav_state = {
                        'position': u_target.copy(),
                        'orientation': np.array([1.0, 0.0, 0.0, 0.0]),
                        'has_collided': False
                    }
                    break

        cur1_pos = uav_state['position']

        pos2_now = None
        max_retries = 3
        for retry in range(max_retries):
            try:
                pos2_now = self.get_object_position()
                if pos2_now is not None:
                    break
            except Exception:
                if retry < max_retries - 1:
                    time.sleep(0.5)
                    continue
        if pos2_now is None:
            pos2_now = t_target.copy()

        jammer_pos_now = None
        jammer_positions_now_by_id = {}
        if jammer_traj is not None:
            max_retries = 3
            for retry in range(max_retries):
                try:
                    jammer_pos_now = self.get_jammer_position()
                    if jammer_pos_now is not None:
                        break
                except Exception:
                    if retry < max_retries - 1:
                        time.sleep(0.5)
                        continue
            if jammer_pos_now is None and j_target is not None:
                jammer_pos_now = j_target.copy()
        if jammer_trajs_by_id:
            for did, traj in jammer_trajs_by_id.items():
                object_name = self._jammer_object_names_by_id.get(str(did), self.jammer_object_name)
                pos = None
                try:
                    pos = self.get_named_object_position(object_name)
                except Exception:
                    pos = None
                if pos is None and j_targets_by_id is not None:
                    pos = j_targets_by_id.get(str(did))
                if pos is not None:
                    jammer_positions_now_by_id[str(did)] = pos

        traj_num = trajectory_name.replace('trajectory_', '') if 'trajectory_' in trajectory_name else trajectory_name

        next_target_pos_airsim = None
        if i + 1 < len(target_traj):
            next_target_pos_airsim = np.array([
                target_traj[i + 1][0],
                target_traj[i + 1][1],
                target_traj[i + 1][2]
            ])

        next_jammer_pos_airsim = None
        if jammer_traj is not None and i + 1 < len(jammer_traj):
            next_jammer_pos_airsim = np.array([
                jammer_traj[i + 1][0],
                jammer_traj[i + 1][1],
                jammer_traj[i + 1][2]
            ])
        next_jammer_positions_by_id = None
        if jammer_trajs_by_id:
            next_jammer_positions_by_id = {}
            for did, traj in jammer_trajs_by_id.items():
                if traj is not None and i + 1 < len(traj):
                    next_jammer_positions_by_id[str(did)] = np.array([
                        traj[i + 1][0], traj[i + 1][1], traj[i + 1][2]
                    ])

        if uav_state.get('has_collided', False):
            pbar.set_postfix_str("", refresh=True)
            tqdm.write("", file=sys.stderr)
            safe_log(f"⚠ Collision detected in trajectory {traj_num} at step {i}", scene_id=self.scene_id)
            raise RuntimeError(f"collision: trajectory {traj_num} step {i}")

        distance = np.linalg.norm(cur1_pos - pos2_now) if pos2_now is not None else 0.0
        jammer_distance = np.linalg.norm(cur1_pos - jammer_pos_now) if jammer_pos_now is not None else None

        try:
            postfix = (
                f"i={i}/{num_steps-1} "
                f"D1=({cur1_pos[0]:.1f},{cur1_pos[1]:.1f},{-cur1_pos[2]:.1f}) "
                f"T=({pos2_now[0]:.1f},{pos2_now[1]:.1f},{-pos2_now[2]:.1f}) "
                f"dist={distance:.1f}m"
            )
            if jammer_pos_now is not None:
                postfix += f" J=({jammer_pos_now[0]:.1f},{jammer_pos_now[1]:.1f},{-jammer_pos_now[2]:.1f}) jdist={jammer_distance:.1f}m"
            pbar.set_postfix_str(postfix, refresh=False)
        except Exception:
            pass

        if save_dataset:
            rgb_img, depth_img = self.get_camera_images()
            self.save_frame_data(i, rgb_img, depth_img, dataset_dir)

        if target_trajectory_airsim is not None and pos2_now is not None:
            target_trajectory_airsim.append({
                "x": float(pos2_now[0]),
                "y": float(pos2_now[1]),
                "z": float(-pos2_now[2])
            })

        if jammer_trajectory_airsim is not None and jammer_pos_now is not None:
            jammer_trajectory_airsim.append({
                "x": float(jammer_pos_now[0]),
                "y": float(jammer_pos_now[1]),
                "z": float(-jammer_pos_now[2])
            })
        if jammer_trajectories_airsim_by_id is not None and jammer_positions_now_by_id:
            for did, pos in jammer_positions_now_by_id.items():
                if did not in jammer_trajectories_airsim_by_id:
                    jammer_trajectories_airsim_by_id[did] = []
                jammer_trajectories_airsim_by_id[did].append({
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(-pos[2]),
                })

        self._append_trajectory_data(
            i,
            uav_state,
            cur1_pos,
            pos2_now,
            merged_trajectory_data,
            next_target_pos_airsim=next_target_pos_airsim,
        )

    def _save_optional_object_trajectory(self, dataset_path, file_name, key_name, trajectory_airsim, num_frames):
        if trajectory_airsim is None or len(trajectory_airsim) == 0:
            return
        traj_path = dataset_path / file_name
        temp_path = dataset_path / f"{file_name}.tmp"
        trajectory_save = []
        for pos in trajectory_airsim:
            if isinstance(pos, dict):
                z_value = pos["z"]
                if z_value < 0:
                    z_value = -z_value
                trajectory_save.append({
                    "x": float(pos["x"]),
                    "y": float(pos["y"]),
                    "z": float(z_value),
                })
            else:
                z_value = pos[2]
                if z_value < 0:
                    z_value = -z_value
                trajectory_save.append({
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(z_value),
                })
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump({
                "num_frames": num_frames,
                key_name: trajectory_save,
            }, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(traj_path)

    def _save_multi_jammer_trajectories(self, dataset_path, jammer_trajectories_airsim_by_id):
        if not jammer_trajectories_airsim_by_id:
            return
        traj_path = dataset_path / "jammer_trajectories.json"
        temp_path = dataset_path / "jammer_trajectories.json.tmp"
        payload = {}
        asset_payload = {}
        max_frames = 0
        for did, series in jammer_trajectories_airsim_by_id.items():
            if series is None or len(series) == 0:
                continue
            converted = []
            for pos in series:
                if isinstance(pos, dict):
                    z_value = pos["z"]
                    if z_value < 0:
                        z_value = -z_value
                    converted.append({
                        "x": float(pos["x"]),
                        "y": float(pos["y"]),
                        "z": float(z_value),
                    })
                else:
                    z_value = pos[2]
                    if z_value < 0:
                        z_value = -z_value
                    converted.append({
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "z": float(z_value),
                    })
            if converted:
                did = str(did)
                payload[did] = converted
                asset_name = self._jammer_asset_names_by_id.get(did)
                if asset_name:
                    asset_payload[did] = asset_name
                if len(converted) > max_frames:
                    max_frames = len(converted)
        if not payload:
            return
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump({
                "num_jammers": len(payload),
                "num_frames": max_frames,
                "jammer_asset_names": asset_payload,
                "jammer_trajectories_airsim": payload,
            }, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(traj_path)

    def _save_trajectory_files(self, dataset_dir, num_steps, selected_uav_name,
                              merged_trajectory_data, save_dataset, target_trajectory_airsim=None,
                              planer_target_num_frames=None, selected_jammer_name=None,
                              jammer_trajectory_airsim=None, planer_jammer_num_frames=None,
                              jammer_trajectories_airsim_by_id=None):
        if not save_dataset:
            return
        dataset_path = Path(dataset_dir)
        try:
            uav_traj_path = dataset_path / "uav_trajectory.json"
            temp_uav_path = dataset_path / "uav_trajectory.json.tmp"
            payload = {
                "num_frames": num_steps,
                "target_asset_name": selected_uav_name,
                "trajectory": merged_trajectory_data,
            }
            if isinstance(selected_jammer_name, dict):
                payload["jammer_asset_names"] = selected_jammer_name
                primary_id = str(self._primary_jammer_id) if self._primary_jammer_id is not None else None
                if primary_id is not None and primary_id in selected_jammer_name:
                    payload["jammer_asset_name"] = selected_jammer_name[primary_id]
            elif selected_jammer_name is not None:
                payload["jammer_asset_name"] = selected_jammer_name
            with open(temp_uav_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            temp_uav_path.replace(uav_traj_path)
        except Exception as e:
            safe_log(f"⚠ Failed to query UAV state: {e}", scene_id=self.scene_id)
            raise

        try:
            target_num_frames = planer_target_num_frames if planer_target_num_frames is not None else (len(target_trajectory_airsim) if target_trajectory_airsim is not None else 0)
            self._save_optional_object_trajectory(dataset_path, "target_trajectory.json", "target_trajectory_airsim", target_trajectory_airsim, target_num_frames)
        except Exception as e:
            safe_log(f"⚠ Failed to serialize target trajectory: {e}", scene_id=self.scene_id)

        # Legacy single-jammer file is intentionally disabled.
        # Keep only multi-jammer export: jammer_trajectories.json
        try:
            self._save_multi_jammer_trajectories(dataset_path, jammer_trajectories_airsim_by_id)
        except Exception as e:
            safe_log(f"⚠ Failed to serialize multi jammer trajectories: {e}", scene_id=self.scene_id)

    def _try_load_planner_positions(self, trajectory_file, suffixes, keys):
        positions_airsim = None
        num_frames = None
        try:
            trajectory_path = Path(trajectory_file)
            base_name = self._extract_trajectory_base_name(trajectory_path)
            candidate_files = []
            for suffix in suffixes:
                candidate_files.append(trajectory_path.parent / f"{base_name}_{suffix}.json")
            for suffix in suffixes:
                candidate_files.append(trajectory_path.parent / f"{suffix}_trajectory.json")

            for candidate in candidate_files:
                if not candidate.exists():
                    continue
                with open(candidate, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key in keys:
                    if key in data and isinstance(data[key], list):
                        traj_raw = data[key]
                        num_frames = len(traj_raw)
                        positions_airsim = []
                        dataset_format = key.endswith('_airsim')
                        for pos in traj_raw:
                            if isinstance(pos, dict) and all(k in pos for k in ('x', 'y', 'z')):
                                x = float(pos['x'])
                                y = float(pos['y']) if dataset_format else float(-pos['y'])
                                z = float(-pos['z'])
                                positions_airsim.append({"x": x, "y": y, "z": z})
                            elif isinstance(pos, list) and len(pos) >= 3:
                                x = float(pos[0])
                                y = float(pos[1]) if dataset_format else float(-pos[1])
                                z = float(-pos[2])
                                positions_airsim.append({"x": x, "y": y, "z": z})
                        if positions_airsim:
                            return num_frames, positions_airsim
        except Exception as e:
            safe_log(f"⚠ planner positions load failed: {e}", scene_id=self.scene_id)
        return num_frames, positions_airsim

    def execute_trajectory(self, trajectory_file, dataset_base_dir="/mnt/Data20T/ysq/OurVLN/Dataset", save_dataset=True, skip_hover=False, trajectory_index=None, total_trajectories=None, max_retries=5, jump_threshold=10.0):
        if not os.path.exists(trajectory_file):
            print(f"Starting trajectory execution: {trajectory_file}")
            return

        uav_traj, target_traj, jammer_traj = self.load_trajectory_bundle(trajectory_file)

        if save_dataset:
            trajectory_name = Path(trajectory_file).stem
            if trajectory_name.endswith('_uav'):
                trajectory_name = trajectory_name[:-4]
            elif trajectory_name.endswith('_target'):
                trajectory_name = trajectory_name[:-7]
            elif trajectory_name.endswith('_jammer'):
                trajectory_name = trajectory_name[:-7]

            dataset_path = Path(dataset_base_dir) / self.scene_id / trajectory_name
            rgb_dir = dataset_path / "rgb"
            uav_json_file = dataset_path / "uav_trajectory.json"
            target_json_file = dataset_path / "target_trajectory.json"
            jammer_multi_json_file = dataset_path / "jammer_trajectories.json"

            num_steps = min(len(uav_traj), len(target_traj), len(jammer_traj) if jammer_traj is not None else 10**18)
            # Always require jammer_trajectories.json whenever jammer trajectory exists
            # (single jammer or multiple jammers).
            need_multi_jammer_json = bool(jammer_traj is not None and len(jammer_traj) > 0)
            jsons_complete = (
                uav_json_file.exists()
                and target_json_file.exists()
                and ((not need_multi_jammer_json) or jammer_multi_json_file.exists())
            )

            if rgb_dir.exists() and jsons_complete:
                existing_frames = []
                for frame_file in rgb_dir.glob("frame_*.png"):
                    try:
                        frame_num = int(frame_file.stem.split('_')[1])
                        existing_frames.append(frame_num)
                    except Exception:
                        continue
                if len(existing_frames) >= num_steps:
                    expected_frames = set(range(num_steps))
                    saved_frames = set(existing_frames)
                    if expected_frames.issubset(saved_frames):
                        safe_log(f"⏭ [{self.scene_id}] Skipping trajectory {trajectory_name}; dataset already complete", scene_id=self.scene_id)
                        return
            elif rgb_dir.exists() and uav_json_file.exists() and (
                (need_multi_jammer_json and (not jammer_multi_json_file.exists()))
            ):
                pass

        try:
            if self.client is not None:
                try:
                    self.client.enableApiControl(False, vehicle_name=self.uav_vehicle_name)
                except Exception:
                    pass
        except Exception as e:
            safe_log(f"⚠ Processing warning: {e}", scene_id=self.scene_id)

        self._abnormal_jumps = []

        retry_count = 0
        while retry_count <= max_retries:
            try:
                return self._execute_trajectory_internal(
                    trajectory_file,
                    dataset_base_dir,
                    save_dataset,
                    skip_hover,
                    trajectory_index,
                    total_trajectories,
                    uav_traj,
                    target_traj,
                    jammer_traj,
                    jump_threshold=jump_threshold,
                )
            except RuntimeError as e:
                error_msg = str(e)
                if "collision" in error_msg.lower() or "distance>=10m" in error_msg.lower() or "abnormal jump" in error_msg.lower():
                    retry_count += 1
                    if retry_count <= max_retries:
                        safe_log(f"🔄 Retry {retry_count}: {error_msg}", scene_id=self.scene_id)
                        try:
                            self._cleanup_after_execution(skip_hover=True)
                        except Exception:
                            pass
                        try:
                            trajectory_name = Path(trajectory_file).stem
                            if trajectory_name.endswith('_uav'):
                                trajectory_name = trajectory_name[:-4]
                            elif trajectory_name.endswith('_target'):
                                trajectory_name = trajectory_name[:-7]
                            elif trajectory_name.endswith('_jammer'):
                                trajectory_name = trajectory_name[:-7]
                            dataset_path = Path(dataset_base_dir) / self.scene_id / trajectory_name
                            for json_file in ['uav_trajectory.json', 'target_trajectory.json', 'jammer_trajectories.json']:
                                json_path = dataset_path / json_file
                                if json_path.exists():
                                    json_path.unlink()
                        except Exception:
                            pass
                        time.sleep(0.5)
                        continue
                    else:
                        safe_log(f"✗ Execution failed after {max_retries} retries; giving up", scene_id=self.scene_id)
                        return
                else:
                    raise
            except Exception:
                raise

    def _execute_trajectory_internal(self, trajectory_file, dataset_base_dir, save_dataset, skip_hover, trajectory_index, total_trajectories, uav_traj, target_traj, jammer_traj=None, jump_threshold=10.0):
        self._jump_threshold = jump_threshold
        self._abnormal_jumps = []

        selected_uav_name = self._prepare_target_object()
        jammer_trajs_by_id = self._all_jammer_trajectories_airsim if self._all_jammer_trajectories_airsim else None
        if jammer_trajs_by_id is None and jammer_traj is not None and len(jammer_traj) > 0:
            # Normalize single-jammer path to the same multi-jammer export format.
            jammer_trajs_by_id = {"1": jammer_traj}
            if self._primary_jammer_id is None:
                self._primary_jammer_id = "1"
        selected_jammer_name = self._prepare_all_jammer_objects(jammer_trajs_by_id) if jammer_trajs_by_id else (self._prepare_jammer_object() if jammer_traj is not None else None)

        trajectory_name = Path(trajectory_file).stem
        if trajectory_name.endswith('_uav'):
            trajectory_name = trajectory_name[:-4]
        elif trajectory_name.endswith('_target'):
            trajectory_name = trajectory_name[:-7]
        elif trajectory_name.endswith('_jammer'):
            trajectory_name = trajectory_name[:-7]

        planer_target_num_frames, planer_target_positions_airsim = self._try_load_planner_positions(
            trajectory_file,
            suffixes=['target'],
            keys=['target_trajectory', 'target_trajectory_airsim'],
        )
        planer_jammer_num_frames, planer_jammer_positions_airsim = self._try_load_planner_positions(
            trajectory_file,
            suffixes=['jammer', 'interferer', 'disturb', 'decoy'],
            keys=['jammer_trajectory', 'jammer_trajectory_airsim', 'interferer_trajectory', 'interferer_trajectory_airsim'],
        )

        dataset_dir = self._prepare_dataset_directory(trajectory_name, dataset_base_dir, save_dataset)
        self._initialize_simulation(uav_traj, target_traj, jammer_traj=jammer_traj, jammer_trajs_by_id=jammer_trajs_by_id)
        self._reset_collision_state()

        jammer_lengths = []
        if jammer_traj is not None:
            jammer_lengths.append(len(jammer_traj))
        if jammer_trajs_by_id:
            jammer_lengths.extend([len(v) for v in jammer_trajs_by_id.values() if v is not None])
        num_steps = min([len(uav_traj), len(target_traj)] + (jammer_lengths if jammer_lengths else [10**18]))
        merged_trajectory_data = []
        target_trajectory_airsim = []
        jammer_trajectory_airsim = [] if jammer_traj is not None else None
        jammer_trajectories_airsim_by_id = {str(k): [] for k in jammer_trajs_by_id.keys()} if jammer_trajs_by_id else None
        self._prev_frame_data = None

        try:
            if not self.client.isApiControlEnabled(vehicle_name=self.uav_vehicle_name):
                max_retries = 5
                retry_delay = 2
                for attempt in range(max_retries):
                    try:
                        self.client.enableApiControl(True, vehicle_name=self.uav_vehicle_name)
                        self.client.armDisarm(True, vehicle_name=self.uav_vehicle_name)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            error_msg = str(e)
                            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                                time.sleep(retry_delay)
                                continue
                        raise
        except Exception:
            pass

        self._ensure_uav_flying_state()

        progress_position = getattr(self, '_progress_position', None)
        if trajectory_index is not None and total_trajectories is not None:
            desc = f"[{self.scene_id}] traj {trajectory_index}/{total_trajectories}"
        else:
            desc = f"[{self.scene_id}]"

        if progress_position is not None:
            pbar = tqdm(range(0, num_steps), desc=desc, unit="step", position=progress_position, leave=True, file=sys.stderr, dynamic_ncols=True, mininterval=0.1, maxinterval=1.0)
            self._current_pbar = pbar
        else:
            pbar = tqdm(range(0, num_steps), desc=desc if trajectory_index is not None else "trajectory", unit="step", file=sys.stderr, dynamic_ncols=True, mininterval=0.1)
            self._current_pbar = pbar

        for i in pbar:
            try:
                self._process_frame(
                    i,
                    uav_traj,
                    target_traj,
                    trajectory_name,
                    num_steps,
                    save_dataset,
                    dataset_dir,
                    merged_trajectory_data,
                    pbar,
                    target_trajectory_airsim=target_trajectory_airsim,
                    jammer_traj=jammer_traj,
                    jammer_trajectory_airsim=jammer_trajectory_airsim,
                    jammer_trajs_by_id=jammer_trajs_by_id,
                    jammer_trajectories_airsim_by_id=jammer_trajectories_airsim_by_id,
                )
            except RuntimeError as e:
                error_msg = str(e)
                if "collision" in error_msg.lower() or "distance>=10m" in error_msg.lower() or "abnormal jump" in error_msg.lower():
                    safe_log(f"✗ Step {i} failed: {e}", scene_id=self.scene_id)
                    try:
                        pbar.close()
                    except Exception:
                        pass
                    raise
                else:
                    safe_log(f"✗ Step {i} failed: {e}", scene_id=self.scene_id)
                    continue
            except Exception as e:
                safe_log(f"✗ Step {i} failed: {e}", scene_id=self.scene_id)
                import traceback
                import io
                traceback_str = io.StringIO()
                traceback.print_exc(file=traceback_str)
                safe_log(traceback_str.getvalue(), scene_id=self.scene_id)
                continue

        try:
            if pbar is not None:
                pbar.refresh()
                pbar.close()
        except Exception:
            pass
        finally:
            if hasattr(self, '_current_pbar'):
                self._current_pbar = None

        try:
            if num_steps > 0:
                self._last_uav_position = np.array([uav_traj[num_steps - 1][0], uav_traj[num_steps - 1][1], uav_traj[num_steps - 1][2]])
                try:
                    uav_state = self.get_uav_state()
                    self._last_uav_position = uav_state['position']
                except Exception:
                    pass
        except Exception:
            pass

        try:
            target_positions_to_save = planer_target_positions_airsim if planer_target_positions_airsim is not None else target_trajectory_airsim
            jammer_positions_to_save = planer_jammer_positions_airsim if planer_jammer_positions_airsim is not None else jammer_trajectory_airsim
            self._save_trajectory_files(
                dataset_dir,
                num_steps,
                selected_uav_name,
                merged_trajectory_data,
                save_dataset,
                target_positions_to_save,
                planer_target_num_frames=planer_target_num_frames,
                selected_jammer_name=selected_jammer_name,
                jammer_trajectory_airsim=jammer_positions_to_save,
                planer_jammer_num_frames=planer_jammer_num_frames,
                jammer_trajectories_airsim_by_id=jammer_trajectories_airsim_by_id,
            )
        except Exception as e:
            safe_log(f"⚠ Processing warning: {e}", scene_id=self.scene_id)
            import traceback
            traceback.print_exc()

        self._check_final_distance(trajectory_name, uav_traj, target_traj)

        try:
            self._cleanup_after_execution(skip_hover)
        except Exception as e:
            safe_log(f"⚠ Processing warning: {e}", scene_id=self.scene_id)

    def _cleanup_after_execution(self, skip_hover):
        super()._cleanup_after_execution(skip_hover)
        if self.client is not None and self.jammer_enabled:
            try:
                self.client.simDestroyObject(self.jammer_object_name)
            except Exception:
                try:
                    existing_objects = self.client.simListSceneObjects(self.jammer_object_name + ".*")
                    if existing_objects:
                        for obj_name in existing_objects:
                            try:
                                self.client.simDestroyObject(obj_name)
                            except Exception:
                                pass
                except Exception:
                    pass
            # Cleanup additional jammer instances (if multi-jammer mode was used)
            try:
                for did, object_name in (self._jammer_object_names_by_id or {}).items():
                    if object_name == self.jammer_object_name:
                        continue
                    try:
                        self.client.simDestroyObject(object_name)
                    except Exception:
                        try:
                            existing_objects = self.client.simListSceneObjects(object_name + ".*")
                            if existing_objects:
                                for obj_name in existing_objects:
                                    try:
                                        self.client.simDestroyObject(obj_name)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
            except Exception:
                pass
