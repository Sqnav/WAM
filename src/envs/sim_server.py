#!/usr/bin/env python3

import os
import sys
import time
import signal
import traceback
import subprocess
import threading
import argparse
import json
from pathlib import Path

CWD_DIR = Path(__file__).resolve().parent


def pid_exists(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def FromPortGetPid(port: int):
    current_pid = os.getpid()

    try:
        p = subprocess.Popen(
            f"ss -ltnp 'sport = :{port}'",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=True,
        )
        output, _ = p.communicate(timeout=2)
        if output:
            text = output.decode("utf-8", errors="ignore")
            for line in text.splitlines():
                if "pid=" in line:
                    try:
                        pid = int(line.split("pid=")[1].split(",")[0])
                        if pid != current_pid:
                            return pid
                    except Exception:
                        pass
    except Exception:
        pass

    return None


def _is_main_script_pid(pid):
    if pid is None or not isinstance(pid, int):
        return False
    try:
        with open(f'/proc/{pid}/cmdline', 'r') as f:
            cmdline = f.read()
        return 'rerun_abnormal_trajectories.py' in cmdline or 'batch_generate_dataset.py' in cmdline
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False


def KillPid(pid, allow_kill_children=False) -> None:
    if pid is None or not isinstance(pid, int):
        return

    current_pid = os.getpid()
    if pid == current_pid:
        print(f"Warning: attempted to kill current process ({current_pid}); skipping")
        return

    if not allow_kill_children:
        parent_pid = os.getppid()
        if pid == parent_pid:
            print(f"Warning: attempted to kill parent process ({parent_pid}); skipping")
            return

    if allow_kill_children:
        try:
            with open(f'/proc/{pid}/cmdline', 'r') as f:
                cmdline = f.read()
            if 'rerun_abnormal_trajectories.py' in cmdline or 'batch_generate_dataset.py' in cmdline:
                print(f"Warning: attempted to kill main script process ({pid}); skipping")
                return
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass

    max_attempts = 50
    attempt = 0
    while pid_exists(pid):
        try:
            if pid == os.getpid():
                print("Warning: detected attempt to kill current process; skipping")
                break
            if attempt == 0 or (attempt + 1) % 10 == 0:
                suffix = f" (attempt {attempt + 1}/{max_attempts})" if attempt > 0 else ""
                print(f"Killing process {pid}{suffix}")
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            break
        except PermissionError:
            print(f"Warning: permission denied when killing process {pid}")
            break
        except Exception as e:
            print(f"Warning: error killing process {pid}: {e}")
            break
        attempt += 1
        if attempt >= max_attempts:
            print(f"Warning: timeout killing process {pid} after {max_attempts} attempts; skipping")
            break
        time.sleep(0.1)


def KillPorts(ports) -> None:
    def _kill_port(port):
        pid = FromPortGetPid(port)
        KillPid(pid, allow_kill_children=True)

    threads = []
    for port in ports:
        thread = threading.Thread(target=_kill_port, args=(port,), daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


_global_root_path = "/mnt/Data20T/ysq/OurVLN"
_global_port = 30000
_global_gpu_ids = [0]


def create_drones(port: int) -> dict:
    return {
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "ClockSpeed": 10,
        "ViewMode": "NoDisplay",
        "Vehicles": {
            "Drone_1": {"VehicleType": "SimpleFlight"}
        },
        "CameraDefaults": {
            "CaptureSettings": [
                {"ImageType": 0, "Width": 640, "Height": 640, "FOV_Degrees": 90},
                {"ImageType": 2, "Width": 640, "Height": 640, "FOV_Degrees": 90},
            ]
        },
        "ApiServerPort": int(port),
    }


def _get_env_exec_path_dict(root_path: str) -> dict:
    root = Path(root_path).resolve()
    # Try the provided root first, then common nearby roots.
    candidate_roots = [root]
    try:
        cwd_root = Path.cwd().resolve()
        for extra in [cwd_root, cwd_root.parent, cwd_root / "code", root.parent]:
            if extra not in candidate_roots:
                candidate_roots.append(extra)
        for parent in [root.parent, cwd_root.parent]:
            sibling_ourvln = parent / "OurVLN"
            if sibling_ourvln not in candidate_roots:
                candidate_roots.append(sibling_ourvln)
    except Exception:
        pass
    legacy_root = Path("/mnt/Data20T/ysq/OurVLN")
    if legacy_root not in candidate_roots:
        candidate_roots.append(legacy_root)

    result = {}
    for candidate_root in candidate_roots:
        env_dir = candidate_root / "Env"
        if not env_dir.exists():
            continue
        for d in env_dir.iterdir():
            if not d.is_dir():
                continue
            scene_id = d.name
            linux_dir = d / "LinuxNoEditor"
            if not linux_dir.exists():
                continue
            sh_name = f"{scene_id}.sh"
            sh_path = linux_dir / sh_name
            if sh_path.exists():
                result[scene_id] = {
                    "exec_path": str(linux_dir.relative_to(candidate_root)),
                    "bash_name": scene_id,
                    "root_path": str(candidate_root),
                }
        if result:
            break
    return result


class EventHandler(object):
    def __init__(self):
        scene_ports = [int(_global_port) + (i + 1) for i in range(1000)]
        self.scene_ports = scene_ports
        self.scene_used_ports = []
        self.scene_processes = {}
        self._port_lock = threading.Lock()

    def ping(self) -> bool:
        return True

    def _open_scenes(self, ip: str, scen_id_gpu_list: list):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\tSTART open scenes: {[s[0] for s in scen_id_gpu_list]}")

        ports = []
        env_exec_path_dict = _get_env_exec_path_dict(_global_root_path)

        with self._port_lock:
            for scen_id, gpu_id in scen_id_gpu_list:
                if isinstance(scen_id, bytes):
                    scen_id = scen_id.decode('utf-8')

                port = None
                try:
                    import re
                    match = re.search(r'(\d+)', str(scen_id))
                    if match:
                        scene_number = int(match.group(1))
                        port = 30000 + scene_number
                    else:
                        raise ValueError("Failed to extract numeric id from scene_id")
                except Exception:
                    for candidate_port in self.scene_ports:
                        if candidate_port not in self.scene_used_ports:
                            pid = FromPortGetPid(candidate_port)
                            if pid is None:
                                port = candidate_port
                                break
                            else:
                                print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\tPort {candidate_port} is used by pid {pid}; cleaning up...")
                                KillPid(pid, allow_kill_children=True)
                                time.sleep(3)
                                pid = FromPortGetPid(candidate_port)
                                if pid is None:
                                    port = candidate_port
                                    break

                if port is None:
                    raise Exception(f"Failed to allocate port for scene {scen_id}")

                if port in self.scene_used_ports:
                    pid = FromPortGetPid(port)
                    if pid is not None:
                        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\tWarning: scene {scen_id} port {port} used by pid {pid}; cleaning up...")
                        KillPid(pid, allow_kill_children=True)
                        max_port_cleanup_retries = 5
                        port_cleared = False
                        for retry in range(max_port_cleanup_retries):
                            time.sleep(2)
                            pid = FromPortGetPid(port)
                            if pid is None:
                                port_cleared = True
                                break
                            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\tPort {port} still used by pid {pid}; retry {retry+1}/{max_port_cleanup_retries}...")
                            KillPid(pid, allow_kill_children=True)

                        if not port_cleared:
                            pid = FromPortGetPid(port)
                            raise Exception(f"Scene {scen_id} port {port} still in use after cleanup (listening pid {pid})")
                    else:
                        self.scene_used_ports.remove(port)

                ports.append(port)
                if port not in self.scene_used_ports:
                    self.scene_used_ports.append(port)

        choose_env_exe_paths = []
        for scen_id, gpu_id in scen_id_gpu_list:
            if isinstance(scen_id, bytes):
                scen_id = scen_id.decode('utf-8')
            if str(scen_id).lower() == 'none':
                choose_env_exe_paths.append(None)
                continue
            scene_key = str(scen_id)
            env_info = None
            if scene_key in env_exec_path_dict:
                env_info = env_exec_path_dict[scene_key]
            else:
                scene_key_lower = scene_key.lower()
                for map_name, info in env_exec_path_dict.items():
                    if str(map_name).lower() == scene_key_lower:
                        env_info = info
                        break

            if env_info is not None:
                res = os.path.join(env_info.get('root_path', _global_root_path), env_info['exec_path'], env_info['bash_name'] + '.sh')
                choose_env_exe_paths.append(res)
            else:
                prefix_flag = False
                for map_name in env_exec_path_dict.keys():
                    if str(scen_id).lower().startswith(str(map_name).lower()):
                        prefix_flag = True
                        env_info = env_exec_path_dict[map_name]
                        res = os.path.join(env_info.get('root_path', _global_root_path), env_info['exec_path'], env_info['bash_name'] + '.sh')
                        choose_env_exe_paths.append(res)
                        break
                if not prefix_flag:
                    raise KeyError(f"Scene executable not found: {scen_id}")

        p_s = []
        for index, (scen_id, gpu_id) in enumerate(scen_id_gpu_list):
            airsim_settings = create_drones(int(ports[index]))
            airsim_settings_write_content = json.dumps(airsim_settings, indent=2, ensure_ascii=False)
            settings_dir = CWD_DIR / 'settings' / str(ports[index])
            settings_dir.mkdir(parents=True, exist_ok=True)
            with open(settings_dir / 'settings.json', 'w', encoding='utf-8') as dump_f:
                dump_f.write(airsim_settings_write_content)

            if choose_env_exe_paths[index] is None:
                p_s.append(None)
                continue

            env_path = choose_env_exe_paths[index]
            if not os.path.exists(env_path):
                raise Exception(f"Scene script not found: {env_path}")
            if not os.access(env_path, os.X_OK):
                try:
                    os.chmod(env_path, 0o755)
                except Exception as e:
                    raise Exception(f"Scene script not executable: {env_path}; failed to chmod +x: {e}")

            vulkan_gpu_id = gpu_id
            settings_path = str(settings_dir / 'settings.json')
            import getpass
            current_user = getpass.getuser()
            if current_user == 'root':
                subprocess_execute = f'runuser -l ysq -c "export CUDA_VISIBLE_DEVICES={gpu_id} && bash {env_path} -RenderOffscreen -NoSound -NoVSync -GraphicsAdapter={vulkan_gpu_id} -settings=\'{settings_path}\'"'
            else:
                subprocess_execute = f'CUDA_VISIBLE_DEVICES={gpu_id} bash {env_path} -RenderOffscreen -NoSound -NoVSync -GraphicsAdapter={vulkan_gpu_id} -settings=\'{settings_path}\''

            time.sleep(1)
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\tScene {scen_id} start cmd: {subprocess_execute}")

            log_dir = CWD_DIR / 'logs'
            log_dir.mkdir(exist_ok=True)
            log_file_path = log_dir / f'scene_{scen_id}_{ports[index]}.log'

            try:
                with self._port_lock:
                    old_info = self.scene_processes.get(scen_id)
                    if old_info is not None:
                        old_p = old_info.get('process')
                        try:
                            if old_p is not None and old_p.poll() is None:
                                print(f"Closing old scene process: {scen_id}, pid={old_p.pid}, port={old_info.get('port')}")
                                try:
                                    os.killpg(os.getpgid(old_p.pid), signal.SIGKILL)
                                except Exception:
                                    try:
                                        old_p.kill()
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                env = os.environ.copy()
                with open(log_file_path, 'w', encoding='utf-8') as log_file:
                    p = subprocess.Popen(
                        subprocess_execute,
                        stdin=None,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        shell=True,
                        env=env,
                        start_new_session=True,
                    )
                p_s.append(p)
                with self._port_lock:
                    self.scene_processes[scen_id] = {'process': p, 'port': ports[index], 'gpu_id': gpu_id}
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\tScene {scen_id} starting, GPU {gpu_id}, log: {log_file_path}")
            except Exception as e:
                raise Exception(f"Failed to start scene: {e}")

        initial_wait = 5
        max_checks = 6
        time.sleep(initial_wait)
        for check_count in range(max_checks):
            failed = []
            for index, (scen_id, gpu_id) in enumerate(scen_id_gpu_list):
                if p_s[index] is not None and p_s[index].poll() is not None:
                    exit_code = p_s[index].returncode
                    failed.append((scen_id, f"Scene {scen_id} process exited (code: {exit_code})"))
            if failed:
                error_details = "; ".join([f"{s}: {e}" for s, e in failed])
                return False, f"Scene start failed: {error_details}"
            if check_count < max_checks - 1:
                time.sleep(5)

        print(f"Scenes started: {ip}, ports: {ports}")
        return True, (ip, ports)

    def reopen_scenes(self, ip: str, scen_id_gpu_list: list):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\tSTART reopen_scenes")
        try:
            if isinstance(ip, bytes):
                ip = ip.decode('utf-8')
            decoded_list = []
            for scen_id, gpu_id in scen_id_gpu_list:
                sid = scen_id.decode('utf-8') if isinstance(scen_id, bytes) else scen_id
                decoded_list.append((sid, gpu_id))
            result = self._open_scenes(ip, decoded_list)
        except Exception as e:
            error_msg = str(e)
            print(error_msg)
            exe_type, exe_value, exe_traceback = sys.exc_info()
            exe_info_list = traceback.format_exception(exe_type, exe_value, exe_traceback)
            tracebacks = ''.join(exe_info_list)
            print('traceback:', tracebacks)
            result = False, f"{error_msg}\nTraceback:\n{tracebacks}"
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\tEND reopen_scenes")
        return result

    def close_scenes(self, ip: str, scene_ids: list = None) -> bool:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\tSTART close_scenes (scene_ids={scene_ids})")
        try:
            with self._port_lock:
                to_remove = []
                if scene_ids is not None and len(scene_ids) > 0:
                    target_set = set(s.strip() for s in scene_ids if isinstance(s, str) and s.strip())
                    for scen_id, info in list(self.scene_processes.items()):
                        if scen_id not in target_set:
                            continue
                        to_remove.append(scen_id)
                        p = info.get('process')
                        port = info.get('port')
                        try:
                            if p is not None and p.poll() is None:
                                print(f"Closing scene process: {scen_id}, pid={p.pid}, port={port}")
                                try:
                                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                                except Exception:
                                    try:
                                        p.kill()
                                    except Exception:
                                        pass
                        except Exception as e:
                            print(f"Failed to close scene {scen_id}: {e}")
                    for scen_id in to_remove:
                        self.scene_processes.pop(scen_id, None)
                    self.scene_used_ports = [info.get('port') for info in self.scene_processes.values() if info.get('port') is not None]
                else:
                    for scen_id, info in list(self.scene_processes.items()):
                        p = info.get('process')
                        port = info.get('port')
                        try:
                            if p is not None and p.poll() is None:
                                print(f"Closing scene process: {scen_id}, pid={p.pid}, port={port}")
                                try:
                                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                                except Exception:
                                    try:
                                        p.kill()
                                    except Exception:
                                        pass
                        except Exception as e:
                            print(f"Failed to close scene {scen_id}: {e}")
                    self.scene_processes.clear()
                    self.scene_used_ports = []
            result = True
        except Exception as e:
            print(e)
            result = False
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\tEND close_scenes")
        return result


def serve_background(server, daemon=False):
    def _start_server(server):
        server.start()
        server.close()
    t = threading.Thread(target=_start_server, args=(server,))
    t.daemon = daemon
    t.start()
    return t


def serve(root_path, port, gpu_ids, daemon=False):
    global _global_root_path, _global_port, _global_gpu_ids
    _global_root_path = root_path
    _global_port = port
    _global_gpu_ids = gpu_ids

    try:
        import msgpackrpc
        server = msgpackrpc.Server(EventHandler())
        addr = msgpackrpc.Address('127.0.0.1', port)
        server.listen(addr)
        thread = serve_background(server, daemon)
        return addr, server, thread
    except Exception as err:
        print(f"Error: {err}")
        if "Address already in use" in str(err) or "errno 98" in str(err).lower():
            print(f"Port {port} is already in use. Stop the existing SimServerTool or choose another port (--port).")
            print(f"  lsof -i :{port} or ss -tlnp | grep {port}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='1')
    parser.add_argument('--port', type=int, default=30000, help='server port')
    # /code/src/envs -> project root should be ../../../
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    parser.add_argument('--root_path', type=str, default=default_root, help='root dir for env path')
    args = parser.parse_args()

    gpu_list = [int(g.strip()) for g in str(args.gpus).split(',') if g.strip()]
    if not gpu_list:
        gpu_list = [0]

    try:
        addr, server, thread = serve(args.root_path, args.port, gpu_list)
        print(f"start listening {addr._host}:{addr._port}")
        time.sleep(0.1)
        try:
            thread.join()
        except KeyboardInterrupt:
            print("\nInterrupted; exiting")
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)
