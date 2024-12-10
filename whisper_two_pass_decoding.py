import re
from typing import Union

import jiwer
import numpy as np
import torch
import tqdm
from datasets import load_dataset, Audio, Dataset
from transformers import AutoProcessor, WhisperForConditionalGeneration
from transformers.generation.utils import GenerateOutput


class TwoPassWhisperModel(WhisperForConditionalGeneration):
    def generate(
            self,
            *args,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        output = super().generate(
            *args,
            **kwargs,
        )
        return output


def parse_string_to_objects(s):
    # Regular expression to match the time tokens
    time_pattern = re.compile(r'<\|([\d.]+)\|>')

    # Find all time tokens
    times = time_pattern.findall(s)

    # Split the text using the time tokens to get the text segments
    text_segments = time_pattern.split(s)[1:]  # Ignore the first empty element

    # Create the list of objects with start, end, and text
    objects = []
    for i in range(0, len(times) - 1):
        start_time = float(times[i])
        end_time = float(times[i + 1])
        text = text_segments[2 * i + 1].strip()
        if text:  # Only add if there is some text
            objects.append({
                'start': start_time,
                'end': end_time,
                'text': text
            })

    return objects


def flatten_generated_segments_to_text(segments):
    return " ".join([segment["text"] for segment in segments])


def flatten_generated_segments_to_text_with_timestamps(segments):
    return " ".join([f"<|{segment['start']:.2f}|>{segment['text']}<|{segment['end']:.2f}|>" for segment in segments])


def normalize_text(text):
    # Lowercase the text and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


if __name__ == "__main__":
    SPLIT = "validation"
    PAUSE = 2.0
    MAX_SINGLE_LAN_SEGMENT_LEN = 14.0
    # logging.set_verbosity_debug()
    # logger = logging.get_logger("transformers")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = TwoPassWhisperModel.from_pretrained(
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
        "return_timestamps": True,
    }

    gen_kwargs_local = {
        "max_new_tokens": 1,
        "num_beams": 1,
        "return_timestamps": True,
    }

    # Limit the samples length in the train split to 15s to be able to easily concatenate them
    for dataset in [dataset_en, dataset_cs]:
        dataset[SPLIT] = dataset[SPLIT].filter(
            lambda x: (x["num_samples"] / x['audio']['sampling_rate']) < MAX_SINGLE_LAN_SEGMENT_LEN)

    max_number_of_samples = max(len(dataset_en[SPLIT]), len(dataset_cs[SPLIT]))

    # create new dataset by concatenating samples from two languages
    gt = []
    hyp = []
    for index, (en_sample, cz_sample) in tqdm.tqdm(enumerate(zip(dataset_en[SPLIT], dataset_cs[SPLIT]))):
        concatenated_sample = torch.from_numpy(np.concatenate(
            [en_sample["audio"]["array"], np.zeros(round(en_sample["audio"]["sampling_rate"] * PAUSE)),
             cz_sample["audio"]["array"]]))
        inputs = processor(concatenated_sample,
                           sampling_rate=en_sample["audio"]["sampling_rate"],
                           return_tensors="pt",
                           return_attention_mask=True).to(device, dtype=torch_dtype)
        gt_text = f"<|en|><|0.00|>{en_sample['transcription']}<|{round(en_sample['num_samples'] / 16_000, 2)}|> <|cs|><|{round(en_sample['num_samples'] / 16_000, 2) + PAUSE}|>{cz_sample['transcription']}<|{round(en_sample['num_samples'] / 16_000, 2) + PAUSE + round(cz_sample['num_samples'] / 16_000, 2)}|>"
        pred_ids = model.generate(**inputs, **gen_kwargs)
        generated_text = processor.decode(pred_ids[0], skip_special_tokens=True)
        generated_text_with_timestamps = processor.decode(pred_ids[0], skip_special_tokens=True,
                                                          decode_with_timestamps=True)
        segments = parse_string_to_objects(generated_text_with_timestamps)
        global_lang_id = pred_ids[0, 1]
        # Run second pass decoding to detect the language of the segments
        for segment in segments:
            audio_chunk = concatenated_sample[int(segment["start"] * 16_000):int(segment["end"] * 16_000)]
            inputs = processor(audio_chunk, sampling_rate=16_000, return_tensors="pt", return_attention_mask=True).to(
                device, dtype=torch_dtype)
            pred_ids = model.generate(**inputs, **gen_kwargs_local)
            if pred_ids[0, 1] != global_lang_id:
                # Detected segment with different language than the global language, so re-run the decoding
                pred_ids = model.generate(**inputs, **gen_kwargs)
                segment["text"] = processor.decode(pred_ids[0], skip_special_tokens=True)

        gt_text_normalised = processor.tokenizer.decode(processor.tokenizer.encode(gt_text), skip_special_tokens=True)
        gt.append(normalize_text(gt_text_normalised))
        hyp.append(normalize_text(flatten_generated_segments_to_text(segments)))
        print(processor.tokenizer.decode(processor.tokenizer.encode(gt_text), skip_special_tokens=True,
                                         decode_with_timestamps=True))
        print(flatten_generated_segments_to_text_with_timestamps(segments))

    df = Dataset.from_dict({"gt": gt, "hyp": hyp})
    df.to_csv("results_default.csv")
    metrics = jiwer.compute_measures(gt, hyp)
    print(metrics)
