import os

import wandb
from rsl_rl.runners import OnPolicyRunner

from ms_lab.rl import RslRlVecEnvWrapper
from ms_lab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)


class VelocityOnPolicyRunner(OnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save_onnx(self, path):
    """Save the model and training information."""


    normalizer = None
    policy_path = path.split("model")[0]
    filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
    export_velocity_policy_as_onnx(
        self.alg.policy,
        normalizer=normalizer,
        path=policy_path,
        filename=filename,
      )


  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    if self.logger_type in ["wandb"]:

      policy_path = path.split("model")[0]
      filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
      if self.alg.policy.actor_obs_normalization:
        normalizer = self.alg.policy.actor_obs_normalizer
      else:
        normalizer = None
      export_velocity_policy_as_onnx(
        self.alg.policy,
        normalizer=normalizer,
        path=policy_path,
        filename=filename,
      )
      attach_onnx_metadata(
        self.env.unwrapped,
        wandb.run.name,  # type: ignore
        path=policy_path,
        filename=filename,
      )
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
