"""Copy the installed LeRobot trainer while retaining IRA normalization stats."""
from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("lerobot.scripts.lerobot_train")
source = Path(spec.origin).read_text()
old = '''        if not cfg.resume:
            preprocessor_overrides["normalizer_processor"]["stats"] = dataset.meta.stats
            postprocessor_overrides["unnormalizer_processor"]["stats"] = dataset.meta.stats
'''
if source.count(old) != 1:
    raise RuntimeError("LeRobot trainer differs from the audited version; refusing to patch")
source = source.replace(old, '''        # Keep checkpoint statistics when continuing IRA fine-tuning.
        # The saved pre- and postprocessors use matching action units.
''')
source = source.replace("from .lerobot_eval import eval_policy_all", "from lerobot.scripts.lerobot_eval import eval_policy_all")
out = Path(__file__).with_name("so101_ira_train.py")
out.write_text(source)
print(out)
