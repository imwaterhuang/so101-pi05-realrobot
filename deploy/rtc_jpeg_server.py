"""Guided-RTC pi0.5 policy server used for the SO-101 hardware runs.

Extends the LeRobot 0.6.1 async gRPC protocol: decodes 224x224 JPEG images
pre-processed on the client and uses the previous chunk's raw actions as the RTC prefix
(execution_horizon=32, linear prefix weights). Binds to loopback only; reach it
through an SSH tunnel.
"""
import argparse
import logging
from concurrent import futures

import grpc
import torch
from lerobot.async_inference.configs import PolicyServerConfig
from pi05_jpeg_server import decode as raw_observation_to_observation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.transport import services_pb2_grpc


class DryRunRTCServer(PolicyServer):
    def SendPolicyInstructions(self, request, context):
        result = super().SendPolicyInstructions(request, context)
        if self.policy_type != 'pi05':
            raise ValueError('This experimental server only supports pi05')
        if getattr(self.policy.config, 'use_relative_actions', None) is not False:
            raise ValueError('Only the verified absolute-action checkpoint is supported')
        self.policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=32,
                                                  prefix_attention_schedule=RTCAttentionSchedule.LINEAR)
        self.policy.init_rtc_processor()
        self.policy.eval()
        self.logger.info('RTC_DRY_RUN enabled=true execution_horizon=32')
        return result

    def _predict_action_chunk(self, observation_t):
        raw = observation_t.get_observation()
        rtc_info = raw.pop('_rtc_test', None)
        if not isinstance(rtc_info, dict):
            raise ValueError('Missing RTC dry-run metadata')
        delay = int(rtc_info['delay_steps'])
        guidance = rtc_info.get('guidance', 'linear')
        if guidance not in ('linear', 'ones'):
            raise ValueError('Unknown RTC guidance')
        prev = rtc_info.get('prev_raw')
        if delay < 0 or delay > 50:
            raise ValueError('RTC delay outside chunk range')
        prev_tensor = None
        if prev is not None:
            prev_tensor = torch.as_tensor(prev, dtype=torch.float32, device=self.device)
            if prev_tensor.ndim != 2 or prev_tensor.shape[1] != 6 or not torch.isfinite(prev_tensor).all():
                raise ValueError('Invalid normalized prior action block')
            prev_tensor = prev_tensor[:32]
        horizon = len(prev_tensor) if prev_tensor is not None else 0
        # With LINEAR, start == end gives unit weights only on the valid prefix.
        strong_steps = horizon if guidance == 'ones' else min(delay, horizon)
        obs = raw_observation_to_observation(raw, self.lerobot_features, self.policy_image_features)
        obs = self.preprocessor(obs)
        self.last_processed_obs = observation_t
        output = self.policy.predict_action_chunk(obs, inference_delay=strong_steps, execution_horizon=horizon, prev_chunk_left_over=prev_tensor)
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[-1] != 6:
            raise ValueError(f'Unexpected policy output shape {tuple(output.shape)}')
        output = output[:, :self.actions_per_chunk, :]
        original = output.squeeze(0).detach().cpu()
        processed = torch.stack([self.postprocessor(output[:, i, :]) for i in range(output.shape[1])], dim=1).squeeze(0).detach().cpu()
        if not torch.isfinite(original).all() or not torch.isfinite(processed).all():
            raise ValueError('Nonfinite policy output')
        timed = self._time_action_chunk(observation_t.get_timestamp(), list(processed), observation_t.get_timestep())
        for action, original_step in zip(timed, original):
            action.rtc_guidance = guidance
            action.rtc_test_raw = original_step
            action.rtc_test_applied = prev is not None
        self.logger.info('RTC_DRY_RUN observation=%s delay=%s guided=%s', observation_t.get_timestep(), delay, prev is not None)
        return timed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8601)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = PolicyServerConfig(host='127.0.0.1', port=args.port, fps=30, inference_latency=0.0, obs_queue_timeout=2)
    instance = DryRunRTCServer(config)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(instance, server)
    if server.add_insecure_port(f'127.0.0.1:{args.port}') == 0:
        raise RuntimeError('Unable to bind server')
    server.start()
    print(f'RTC_DRY_RUN server listening on 127.0.0.1:{args.port}', flush=True)
    server.wait_for_termination()

if __name__ == '__main__':
    main()
