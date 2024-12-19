import argparse
import json
from pathlib import Path
import random

from datasets import load_dataset
import jiwer
import numpy as np
import torch
from datasets import load_dataset, Audio, Dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, set_seed


def normalize_text(processor, text):
    # preserves diacritics
    # removes punctuation and performs lowercasing
    return processor.tokenizer.basic_normalize(text)


def load_sdaia(sampling_rate):
    sdaia_path = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/node5/code/Users/btalafha/whisper_hack/scc/cs_only_segments.json'
    sdaia = load_dataset('json', data_files=sdaia_path)

    def update_path(utt):
        utt['path'] = utt['path'].replace("/tmp/SCC/", "/mnt/batch/tasks/shared/LS_root/mounts/clusters/node5/code/Users/btalafha/whisper_hack/scc/")
        return utt
    sdaia = sdaia.map(update_path)
    sdaia = sdaia.rename_column("file", "audio")
    sdaia = sdaia.cast_column("audio", Audio(sampling_rate))
    sdaia = sdaia.rename_column("ref", "transcription")
    # 2 columns: ref, file -> transcription, audio
    return sdaia


def get_parser():
    parser = argparse.ArgumentParser(
        description="Whisper for English-X codeswitching"
    )
    parser.add_argument(
        "--lang_code",
        default="cs_cz",
        required=True,
        choices=["cs_cz", "ar_eg", "sdaia"],
        help="FLEURS language code. lang_locale. Exception: sdaia"
    )
    return parser


def evaluate_sdaia(model, processor,
                   sdaia,
                   device, seed=11711,
                   pause=1.0,
                   run=0):
    # fix the seed
    set_seed(seed)

    gen_kwargs = {
        "max_new_tokens": 440,
        "num_beams": 1,
        "return_timestamps": True,
        "repetition_penalty": 1.2
    }

    # create new dataset by concatenating samples from two languages
    gt = []
    hyp = []
    for index, sample in enumerate(sdaia):
        inputs = processor(sample,
                           sampling_rate=sample["audio"]["sampling_rate"],
                           return_tensors="pt",
                           return_attention_mask=True).to(device, dtype=torch_dtype)
        # skip_special_tokens=True - all special tokens are removed during evaluation
        gt_text = f"<|0.00|>{sample["transcription"]}<|{round(sample['num_samples'] / 16_000, 2)}|>"
        pred_ids = model.generate(**inputs, **gen_kwargs)
        generated_text = processor.decode(pred_ids[0], skip_special_tokens=True)
        # this removes the special tokens for the ground truth
        gt_text_normalised = processor.tokenizer.decode(processor.tokenizer.encode(gt_text), skip_special_tokens=True)

        # add timestamps for visualization, use pre-normalized text
        print("gold", processor.tokenizer.decode(
            processor.tokenizer.encode(gt_text),
            skip_special_tokens=True,
            decode_with_timestamps=True
        ))
        print("hyp", processor.decode(
            pred_ids[0],
            skip_special_tokens=True,
            decode_with_timestamps=True
        ))
        print()

        gt_text_normalised = normalize_text(processor, gt_text_normalised)
        generated_text = normalize_text(processor, generated_text)

        gt.append(gt_text_normalised)
        hyp.append(generated_text)

    Path(f"results/sdaia/").mkdir(parents=True, exist_ok=True)
    df = Dataset.from_dict({"gt": gt, "hyp": hyp})
    df.to_csv(f"results/sdaia/default_{run}.csv")
    metrics = jiwer.process_words(gt, hyp)
    print(metrics.wer)
    return metrics.wer


def evaluate(model, processor,
             dataset_en, dataset_xx, lang,
             device, seed=11711,
             pause=1.0,
             run=0):
    # fix the seed
    set_seed(seed)

    min_number_of_samples = min(len(dataset_en), len(dataset_xx))
    gen_kwargs = {
        "max_new_tokens": 440,
        "num_beams": 1,
        "return_timestamps": True,
        "repetition_penalty": 1.2
    }

    # create new dataset by concatenating samples from two languages
    gt = []
    hyp = []
    for index, (en_sample, xx_sample) in enumerate(zip(dataset_en, dataset_xx)):
        concatenated_sample = torch.from_numpy(np.concatenate(
            [en_sample["audio"]["array"], np.zeros(round(en_sample["audio"]["sampling_rate"] * PAUSE)),
             xx_sample["audio"]["array"]]))
        inputs = processor(concatenated_sample,
                           sampling_rate=en_sample["audio"]["sampling_rate"],
                           return_tensors="pt",
                           return_attention_mask=True).to(device, dtype=torch_dtype)
        # skip_special_tokens=True - all special tokens are removed during evaluation
        gt_text = f"<|en|><|0.00|>{en_sample['transcription']}<|{round(en_sample['num_samples'] / 16_000, 2)}|> <|{lang}|><|{round(en_sample['num_samples'] / 16_000, 2) + PAUSE}|>{xx_sample['transcription']}<|{round(en_sample['num_samples'] / 16_000, 2) + PAUSE + round(xx_sample['num_samples'] / 16_000, 2)}|>"
        pred_ids = model.generate(**inputs, **gen_kwargs)
        generated_text = processor.decode(pred_ids[0], skip_special_tokens=True)
        # this removes the special tokens for the ground truth
        gt_text_normalised = processor.tokenizer.decode(processor.tokenizer.encode(gt_text), skip_special_tokens=True)

        # add timestamps for visualization, use pre-normalized text
        print("gold", processor.tokenizer.decode(
            processor.tokenizer.encode(gt_text),
            skip_special_tokens=True,
            decode_with_timestamps=True
        ))
        print("hyp", processor.decode(
            pred_ids[0],
            skip_special_tokens=True,
            decode_with_timestamps=True
        ))
        print()

        gt_text_normalised = normalize_text(processor, gt_text_normalised)
        generated_text = normalize_text(processor, generated_text)

        gt.append(gt_text_normalised)
        hyp.append(generated_text)

    Path(f"results/{lang}/").mkdir(parents=True, exist_ok=True)
    df = Dataset.from_dict({"gt": gt, "hyp": hyp})
    df.to_csv(f"results/{lang}/default_{run}.csv")
    metrics = jiwer.process_words(gt, hyp)
    print(metrics.wer)
    return metrics.wer


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    LANG_CODE = args.lang_code
    LANG = LANG_CODE.split("_")[0]

    SPLIT = "validation"
    PAUSE = 1.0
    MAX_SINGLE_LAN_SEGMENT_LEN = 6.0

    device = "cuda:4" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    if LANG_CODE == "sdaia":
        sdaia = load_sdaia(processor.feature_extractor.sampling_rate)
        # Limit the samples length in the train split to 15s to be able to easily concatenate them
        sdaia = sdaia.filter(
            lambda x: (x["num_samples"] / x['audio']['sampling_rate']) < MAX_SINGLE_LAN_SEGMENT_LEN)
    else:
        dataset_en = load_dataset("google/fleurs", "en_us", cache_dir="/tmp")
        dataset_xx = load_dataset("google/fleurs", LANG_CODE, cache_dir="/tmp")
        dataset_en = dataset_en.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))
        dataset_xx = dataset_xx.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))

        # Limit the samples length in the train split to 15s to be able to easily concatenate them
        for dataset in [dataset_en, dataset_xx]:
            dataset[SPLIT] = dataset[SPLIT].filter(
                lambda x: (x["num_samples"] / x['audio']['sampling_rate']) < MAX_SINGLE_LAN_SEGMENT_LEN)

    # try 10 seeds
    random.seed(12345)
    runs = []
    for run, seed in enumerate([random.randint(10000, 99999) for _ in range(10)]):
        if LANG_CODE == "sdaia":
            # real codeswitching
            wer = evaluate_sdaia(
                model, processor,
                sdaia,
                device, seed, PAUSE,
                run
            )
        else:
            # synthetic codeswitching
            wer = evaluate(
                model, processor,
                dataset_en[SPLIT], dataset_xx[SPLIT], LANG,
                device, seed, PAUSE,
                run
            )
        runs.append(wer)

    print("WER across", len(runs), "runs:", np.mean(runs), np.std(runs))
    print(runs)
