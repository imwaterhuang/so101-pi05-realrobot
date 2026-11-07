"""Ask the (SSH-forwarded) policy server to load a pi0.5 checkpoint."""
import pickle
import time

import grpc
from lerobot.async_inference.helpers import RemotePolicyConfig, map_robot_keys_to_lerobot_features
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.transport import services_pb2, services_pb2_grpc

import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='pretrained_model dir on the server')
    p.add_argument('--address', default='127.0.0.1:8616')
    p.add_argument('--robot-port', default='/dev/cu.usbmodemXXXX')
    args = p.parse_args()
    cameras = {
        'wrist_left': OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        'desk_view': OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
    }
    robot = SOFollower(SOFollowerRobotConfig(
        port=args.robot_port, id='my_follower', cameras=cameras))
    with grpc.insecure_channel(args.address) as channel:
        stub = services_pb2_grpc.AsyncInferenceStub(channel)
        stub.Ready(services_pb2.Empty(), timeout=15)
        config = RemotePolicyConfig('pi05', args.model, map_robot_keys_to_lerobot_features(robot), 50, 'cuda')
        started = time.perf_counter()
        print('Loading policy on server...', flush=True)
        stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(config)), timeout=300)
        print(f'Server ready in {time.perf_counter() - started:.1f}s', flush=True)

if __name__ == '__main__':
    main()
