import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset, Audio, Dataset
from transformers.utils import logging
import jiwer

if __name__ == "__main__":
    SPLIT = "validation"
    PAUSE = 2.0
    MAX_SINGLE_LAN_SEGMENT_LEN = 14.0
    # logging.set_verbosity_debug()
    # logger = logging.get_logger("transformers")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    dataset_en = load_dataset("google/fleurs", "en_us")
    dataset_cs = load_dataset("google/fleurs", "cs_cz")
    dataset_en = dataset_en.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))
    dataset_cs = dataset_cs.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))

    gen_kwargs = {
        "max_new_tokens": 440,
        "num_beams": 1,
        # "condition_on_prev_tokens": False,
        # "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
        # "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        # "logprob_threshold": -1.0,
        # "no_speech_threshold": 0.6,
        "return_timestamps": True,
    }

    # Limit the samples length in the train split to 15s to be able to easily concatenate them
    for dataset in [dataset_en, dataset_cs]:
        dataset[SPLIT] = dataset[SPLIT].filter(lambda x: (x["num_samples"] / x['audio']['sampling_rate']) < MAX_SINGLE_LAN_SEGMENT_LEN)

    max_number_of_samples = max(len(dataset_en[SPLIT]), len(dataset_cs[SPLIT]))

    # create new dataset by concatenating samples from two languages
    new_items = []
    for index, (en_sample, cz_sample) in enumerate(zip(dataset_en[SPLIT], dataset_cs[SPLIT])):
        concatenated_sample = torch.from_numpy(np.concatenate(
            [en_sample["audio"]["array"], np.zeros(round(en_sample["audio"]["sampling_rate"] * PAUSE)),
             cz_sample["audio"]["array"]]))
        inputs = processor(concatenated_sample,
                           sampling_rate=en_sample["audio"]["sampling_rate"],
                           return_tensors="pt",
                           return_attention_mask=True)
        # new_item = {
        #     "input_features": ,
        #     "text": f"<|en|>{en_sample['transcription']}<|cs|>{cz_sample['transcription']}"
        # }
        gt_text = f"<|en|><|0.00|>{en_sample['transcription']}<|{round(en_sample['num_samples'] / 16_000, 2)}|> <|cs|><|{round(en_sample['num_samples'] / 16_000, 2) + PAUSE}|>{cz_sample['transcription']}<|{round(en_sample['num_samples'] / 16_000, 2) + PAUSE + round(cz_sample['num_samples'] / 16_000, 2)}|>"
        pred_ids = model.generate(**inputs, **gen_kwargs)
        generated_text = processor.decode(pred_ids[0], skip_special_tokens=True)
        gt_text_normalised = processor.tokenizer.decode(processor.tokenizer.encode(gt_text), skip_special_tokens=True)
        print(index, jiwer.wer(gt_text_normalised, generated_text))
        print(processor.tokenizer.decode(processor.tokenizer.encode(gt_text), skip_special_tokens=True,
                                         decode_with_timestamps=True))
        print(processor.decode(pred_ids[0], skip_special_tokens=True, decode_with_timestamps=True))
