"""Accept observations already resized/padded to 224x224 uint8 on the client."""
import torch
import lerobot.async_inference.policy_server as server
from lerobot.async_inference.helpers import make_lerobot_observation, extract_state_from_raw_observation, prepare_image
original = server.raw_observation_to_observation

def prepare(raw, features, image_features):
    if not raw.get('_test_preprocessed_224', False):
        return original(raw, features, image_features)
    mapped = make_lerobot_observation(raw, features)
    out = {'observation.state': extract_state_from_raw_observation(mapped)}
    for key in image_features:
        x = torch.as_tensor(mapped[key])
        if x.dtype != torch.uint8 or tuple(x.shape) != (224, 224, 3):
            raise ValueError('Expected RGB uint8 HWC 224x224')
        out[key] = prepare_image(x.permute(2, 0, 1)).unsqueeze(0)
    if 'task' in raw:
        out['task'] = raw['task']
    return out
server.raw_observation_to_observation = prepare
if __name__ == '__main__':
    server.serve()
