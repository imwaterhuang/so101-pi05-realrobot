"""SO-101 client for cloud pi0.5 inference with guided RTC. Drives motors only with --execute.

Used for the 2026-11-06/07 hardware runs (replace handoff, linear guidance).
It does NOT limit model targets, and torque stays on after exit: supervise the robot
and keep a way to cut power within reach.
"""
import argparse
import math
import pickle
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import cv2
import grpc
import torch
from lerobot.async_inference.helpers import TimedObservation, prepare_image, resize_robot_observation_image
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.policies.common.vla_utils import resize_with_pad_torch
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks
from append_queue import AppendQueue
from joint_trace import JointTrace

JOINTS=('shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos')

def action_dict(action):
    if action.numel()!=6 or not torch.isfinite(action).all():
        raise ValueError('Invalid six-dimensional action')
    return {key:float(action[i]) for i,key in enumerate(JOINTS)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Send unbounded model targets to motors')
    parser.add_argument('--duration',type=float,default=20.0,help='Maximum elapsed seconds after execution starts; 0 = run until Ctrl+C')
    parser.add_argument('--address',default='127.0.0.1:8616')
    parser.add_argument('--robot-port',default='/dev/cu.usbmodemXXXX',help='Follower serial port')
    parser.add_argument('--task',default='Pick up the green ball and place it in the metal bowl')
    # replace: switch at the step actually reached during inference, which lies inside the
    # frozen RTC prefix. append switches at the end of the guided region and caused a
    # ~7 deg jump (pull-back) on every chunk in the 2026-11-06 hardware run.
    parser.add_argument('--handoff', choices=('append', 'replace'), default='replace')
    parser.add_argument('--guidance', choices=('linear', 'ones'), default='linear')
    parser.add_argument('--seam-warn', type=float, default=3.0,
                        help='Print a warning when the handoff jump on any joint exceeds this (deg)')
    parser.add_argument('--seam-abort', type=float, default=None,
                        help='Stop before executing a chunk whose handoff jump exceeds this (deg)')
    args=parser.parse_args()
    for name in ('seam_warn', 'seam_abort'):
        v = getattr(args, name)
        if v is not None and (not math.isfinite(v) or v <= 0): raise ValueError(f'{name} must be > 0')
    if not math.isfinite(args.duration) or args.duration < 0: raise ValueError('duration must be >= 0 (0 = until Ctrl+C)')
    run_limit = math.inf if args.duration == 0 else args.duration
    cams={'wrist_left':OpenCVCameraConfig(index_or_path=0,width=640,height=480,fps=30), 'desk_view':OpenCVCameraConfig(index_or_path=2,width=640,height=480,fps=30)}
    # Cameras are opened by OpenCV below, not through the robot abstraction.
    # Empty robot camera config makes is_connected reflect the motor bus.
    config=SOFollowerRobotConfig(port=args.robot_port,id='my_follower',cameras={},disable_torque_on_disconnect=False,max_relative_target=None,use_degrees=True)
    robot=SOFollower(config);queue=AppendQueue();stop=threading.Event();began=threading.Event();bus_lock=threading.Lock();fault=[];caps=[];stats={'sent':0,'empty':0};worker=None;started_at=[None]
    channel=grpc.insecure_channel(args.address,options=[('grpc.max_send_message_length',67108864),('grpc.max_receive_message_length',67108864)])
    stub=services_pb2_grpc.AsyncInferenceStub(channel);pool=ThreadPoolExecutor(max_workers=1)
    trace = JointTrace(Path(__file__).resolve().parent.parent / 'logs' / 'joint_traces',
                       {**vars(args), 'max_relative_target': None,
                        'joint_order': JOINTS, 'feedback': 'existing observation reads only'})
    print(f'Joint trace: {trace.path}', flush=True)
    last_command = None
    last_command_at = None
    last_sent = None
    def request_loop():
        delays=[]
        stage = "starting"
        try:
            while not stop.is_set():
                if queue.size()>32:
                    stop.wait(.01);continue
                with bus_lock:
                    before = time.monotonic()
                    token, prefix = queue.begin()
                    empty_before = stats['empty']
                prev = torch.stack(prefix) if prefix else None
                obs = {}
                for name,cap,shape in [('wrist_left',caps[0],(3,480,640)),('desk_view',caps[1],(3,600,800))]:
                    ok,frame=cap.read()
                    if not ok or frame is None: raise RuntimeError('Camera failed: '+name)
                    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    x=prepare_image(resize_robot_observation_image(torch.from_numpy(rgb),shape)).unsqueeze(0).permute(0,2,3,1)
                    im=(resize_with_pad_torch(x,224,224)*255).round().clamp(0,255).to(torch.uint8).squeeze(0).numpy()
                    ok,buf=cv2.imencode('.jpg',cv2.cvtColor(im,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
                    if not ok: raise RuntimeError('JPEG encoding failed')
                    obs[name]=buf.tobytes()
                with bus_lock:
                    read_started = time.monotonic()
                    measured = {f'{k}.pos': float(v) for k, v in robot.bus.sync_read('Present_Position').items()}
                    read_finished = time.monotonic()
                    obs.update(measured)
                    trace.record('measured', values=measured,
                                 read_duration_s=read_finished - read_started,
                                 last_command=last_command,
                                 command_age_s=read_finished - last_command_at if last_command_at is not None else None,
                                 tracking_error={k: last_command[k] - measured[k] for k in JOINTS}
                                 if last_command is not None else None)
                obs.update({'task':args.task,'_test_preprocessed_224':True,'_test_jpeg_224':True})
                delay=min(len(prefix),math.ceil(max(delays[-3:])*30)) if delays else 0
                obs['_rtc_test']={'delay_steps':delay,'prev_raw':prev,'guidance':args.guidance}
                ident=time.time_ns()//1000000
                data=pickle.dumps(TimedObservation(timestamp=time.time(),timestep=ident,observation=obs,must_go=True))
                stage = "GetActions / SendObservations"
                pending=pool.submit(stub.GetActions,services_pb2.Empty(),timeout=5)
                stub.SendObservations(send_bytes_in_chunks(data,services_pb2.Observation,silent=True),timeout=5)
                stage = "waiting for GetActions"
                response=pending.result(timeout=5);deadline=time.monotonic()+5
                while not response.data or pickle.loads(response.data)[0].get_timestep()!=ident:
                    if stop.is_set(): return
                    if time.monotonic()>deadline:raise TimeoutError('No matching action')
                    response=stub.GetActions(services_pb2.Empty(),timeout=2)
                elapsed=time.monotonic()-before
                if stop.is_set(): return
                actions=pickle.loads(response.data)
                if len(actions)!=50 or not all(hasattr(a,'rtc_test_raw') for a in actions):raise ValueError('Unexpected RTC response')
                raw=[a.rtc_test_raw.detach().cpu() for a in actions];processed=[a.get_action().detach().cpu() for a in actions]
                for a in processed:action_dict(a)
                if getattr(actions[0], 'rtc_guidance', None) != args.guidance:
                    raise RuntimeError('Server did not confirm the requested guidance mode')
                with bus_lock:
                    if stop.is_set():
                        return
                    remaining = queue.size()
                    consumed = len(prefix) - remaining
                    replace = args.handoff == 'replace'
                    if replace and stats['empty'] != empty_before:
                        raise RuntimeError('Queue underrun during inference; fresh observation required')
                    skip = consumed if replace else len(prefix)
                    tail, first = queue.finish(token, len(prefix), raw, processed, replace=replace)
                    previous = last_sent if tail is None else tail
                    seam = (first - previous).tolist() if previous is not None else None
                    seam_max = max(abs(x) for x in seam) if seam is not None else 0.0
                    # Jump inside the model's own chunk at the switch index (independent of the queue).
                    inner = (processed[skip] - processed[skip - 1]).tolist() if 0 < skip < len(processed) else None
                    # In replace mode the switch must fall inside the frozen prefix (consumed <= delay).
                    outside_frozen = replace and len(prefix) > 0 and consumed > delay
                    if args.seam_abort is not None and seam_max > args.seam_abort:
                        stop.set()  # set while holding the bus lock so the control loop pops nothing new
                        raise RuntimeError(f'Handoff jump {seam_max:.2f} deg exceeds --seam-abort {args.seam_abort}')
                trace.record('chunk', chunk=ident, elapsed_s=elapsed, estimate=delay,
                             consumed=consumed, skip=skip, prefix_length=len(prefix), seam_delta=seam,
                             seam_max=seam_max, inner_jump=inner, outside_frozen=outside_frozen,
                             processed_actions=[action_dict(a) for a in processed])
                print(f'chunk={ident} handoff={args.handoff} guidance={args.guidance} '
                      f'estimate={delay} consumed={consumed} skip={skip} seam_max={seam_max:.2f}', flush=True)
                if seam_max > args.seam_warn:
                    print(f'WARNING seam: chunk={ident} jump={seam_max:.2f} deg seam_delta={seam}', flush=True)
                if outside_frozen:
                    print(f'WARNING delay: consumed={consumed} > estimate={delay}; switch is outside the frozen prefix', flush=True)
                delays.append(elapsed)
                if not began.is_set(): started_at[0]=time.monotonic();began.set()
                print(f'chunk={ident} elapsed={elapsed:.3f}s prefix={len(prefix)} queue={queue.size()}',flush=True)
        except Exception as exc:
            trace.record('request_error', stage=stage, error=repr(exc))
            fault.append(exc);stop.set();print('REQUEST ERROR:', stage, repr(exc), flush=True)
    try:
        print("Checking gRPC connection...", flush=True)
        grpc.channel_ready_future(channel).result(timeout=5)
        for idx in (0,2):
            cap=cv2.VideoCapture(idx,cv2.CAP_AVFOUNDATION);caps.append(cap)
            if not cap.isOpened():raise RuntimeError(f'Camera {idx} unavailable')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        time.sleep(2)
        with bus_lock:
            robot.bus.connect()
            if not robot.bus.is_calibrated:raise RuntimeError('Calibration mismatch')
        worker=threading.Thread(target=request_loop,daemon=True);worker.start()
        print('MODE:', 'MOTORS ENABLED / NO LIMITS' if args.execute else 'DRY RUN / NO MOTOR COMMANDS',flush=True)
        period = 1 / 30
        next_tick = time.monotonic()
        while not stop.is_set():
            tick = time.monotonic()
            if began.is_set():
                deadline = started_at[0] + run_limit
                if tick >= deadline:
                    break
                if tick - next_tick > 0.1:
                    raise RuntimeError('Control loop delayed more than 100 ms')
                with bus_lock:
                    tick = time.monotonic()
                    if stop.is_set() or tick >= deadline:
                        break
                    if tick - next_tick > 0.1:
                        raise RuntimeError('Control loop delayed more than 100 ms')
                    action = queue.pop()
                    if action is None:
                        stats['empty'] += 1
                        trace.record('empty', count=stats['empty'])
                    else:
                        target = action_dict(action)
                        sequence = stats['sent']
                        trace.record('requested', sequence=sequence, values=target)
                        if args.execute:
                            sent = robot.send_action(target)
                            last_command = {k: float(v) for k, v in sent.items()}
                            last_command_at = time.monotonic()
                            trace.record('sent', sequence=sequence, values=last_command,
                                         target_adjustment={k: last_command[k] - target[k] for k in JOINTS})
                        else:
                            trace.record('simulated', sequence=sequence, values=target)
                        last_sent = action
                        stats['sent'] += 1
            # Anchor to this tick's actual dispatch so a late write never creates catch-up ticks.
            next_tick = tick + period
            stop.wait(max(0, next_tick - time.monotonic()))
        if fault:raise RuntimeError(f'Inference worker failed: {fault[0]}')
    except KeyboardInterrupt:
        trace.record('exit_ctrl_c')
        print('Ctrl+C received: stopping (torque stays on).', flush=True)
    except BaseException as exc:
        trace.record('exit_error', error=repr(exc))
        raise
    finally:
        stop.set();channel.close()
        if worker:worker.join(timeout=6)
        if worker and worker.is_alive():
            print('ERROR: worker did not stop; bus left untouched',flush=True)
        elif robot.bus.is_connected:
            with bus_lock:robot.bus.disconnect(disable_torque=False)
        if not worker or not worker.is_alive():
            for cap in caps:
                cap.release()
        pool.shutdown(wait=False,cancel_futures=True)
        trace.close()
        print('Stopped. Sent/consumed:',stats,'Torque was not deliberately disabled.',flush=True)

if __name__=='__main__':main()
