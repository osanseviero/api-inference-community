from typing import Dict

import numpy as np
from app.pipelines import Pipeline
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
import os

class AutomaticSpeechRecognitionPipeline(Pipeline):
    def __init__(self, model_id: str):
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            model_id,
            arg_overrides={"fp16": False},
            cache_dir=os.getenv("HUGGINGFACE_HUB_CACHE"),
        )
        self.model = models[0].cpu()
        self.model.eval()
        cfg["task"].cpu = True
        self.task = task
        S2THubInterface.update_cfg_with_data_cfg(cfg, self.task.data_cfg)

        self.generator =  self.task.build_generator(self.model, cfg)
        self.sampling_rate = 16000


    def __call__(self, inputs: np.array) -> Dict[str, str]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at self.sampling_rate, otherwise 16KHz.
        Return:
            A :obj:`dict`:. The object return should be liked {"text": "XXX"} containing
            the detected langage from the input audio
        """
        sample = S2THubInterface.get_model_input(self.task, inputs)
        text = S2THubInterface.get_prediction(self.task, self.model, self.generator, sample)
        return {"text": text}
