import argparse
import random

import jiwer
import numpy as np
import torch
from datasets import load_dataset, Audio, Dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, set_seed


def get_parser():
    parser = argparse.ArgumentParser(
        description="Whisper with speech in-context learning (Wang et al 2024)"
    )
    parser.add_argument(
        "--speech_context",
        default=True,
        required=True,
        action=argparse.BooleanOptionalAction,
        help="Whether or not to include speech in the context.\
            --speech_context or --no-speech_context"
    )
    return parser


def evaluate(model, processor, dataset_en, dataset_cs,
             device, seed=11711, speech_context=True,
             pause=1.0,
             run=0):
    # fix the seed
    set_seed(seed)

    min_number_of_samples = min(len(dataset_en), len(dataset_cs))
    gen_kwargs = {
        "max_new_tokens": 300, # The length of `decoder_input_ids` equal `prompt_ids` plus special start tokens is 54, and the `max_new_tokens` is 440. Thus, the combined length of `decoder_input_ids` and `max_new_tokens` is: 494. This exceeds the `max_target_positions` of the Whisper model: 448. You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, so that their combined length is less than 448.
        "num_beams": 1,
        "return_timestamps": True,
        "repetition_penalty": 1.2
    }

    # create new dataset by concatenating samples from two languages
    gt = []
    hyp = []
    for index, (en_sample, cz_sample) in enumerate(zip(dataset_en, dataset_cs)):
        # select in-context example
        in_context_index = index
        while in_context_index == index:
            in_context_index = random.randint(0, min_number_of_samples - 1)
        ic_en, ic_cz = dataset_en[in_context_index], dataset_cs[in_context_index]

        # target utterance
        target_audio = np.concatenate([
            en_sample["audio"]["array"],
            np.zeros(round(en_sample["audio"]["sampling_rate"] * pause)),
            cz_sample["audio"]["array"],
        ])
        if SPEECH_CONTEXT:
            audio = np.concatenate([
                # in-context example
                ic_en["audio"]["array"],
                np.zeros(round(ic_en["audio"]["sampling_rate"] * pause)),
                ic_cz["audio"]["array"],
                np.zeros(round(ic_en["audio"]["sampling_rate"] * pause)),
                target_audio
            ])
        else:
            # no speech in the context
            # text-only context
            audio = target_audio

        concatenated_sample = torch.from_numpy(audio)
        inputs = processor(concatenated_sample,
                           sampling_rate=en_sample["audio"]["sampling_rate"],
                           return_tensors="pt",
                           return_attention_mask=True).to(device, dtype=torch_dtype)

        # TODO: what do whisper timestamps look like? am i doing it right?
        # target utterance
        time_ic_en_end = round(ic_en['num_samples'] / 16_000, 2)
        time_ic_cz_start = time_ic_en_end + pause
        time_ic_cz_end = time_ic_cz_start + round(ic_cz['num_samples'] / 16_000, 2)
        # note - in both branches, we omit the <|transcribe|> task token
        if SPEECH_CONTEXT:
            # we manually add special tokens to be consistent with the other branch
            # where Whisper will actually duplicate the tokens
            gt_context = f"<|startoftranscript|><|en|><|cs|><|0.00|>{ic_en['transcription']}<|{time_ic_en_end}|><|{time_ic_cz_start}|>{ic_cz['transcription']}<|{time_ic_cz_end}|>"
        else:
            # add start_of_prev manually
                # no timestamp or lang token though
            # no speech in the context, thus no timestamps
            # note: unlike with the normal transcription, a space is expected in the prompt
            gt_context = f"<|startofprev|> {ic_en['transcription']}{ic_cz['transcription']}<|startoftranscript|><|en|><|cs|><|transcribe|>"

        time_en_start = time_ic_cz_end + pause
        time_en_end = time_en_start + round(en_sample['num_samples'] / 16_000, 2)
        time_cz_start = time_en_end + pause
        time_cz_end = time_cz_start + round(cz_sample['num_samples'] / 16_000, 2)
        # adding the timestamp for visual inspection only
        gt_text = f"<|en|><|{time_en_start}|>{en_sample['transcription']}<|{time_en_start}|> <|cs|><|{time_cz_start}|>{cz_sample['transcription']}<|{time_cz_end}|>"

        # set the prefix (in-context text) with decoder_input_ids
        # add_special_tokens=False because we already manually included <|startoftranscript|><|notimestamps|>
        # thus doing so will duplicate <|startoftranscript|>
        context_token_ids = processor.tokenizer.encode(gt_context, add_special_tokens=False, return_tensors="pt").to(device)
        # note that we manually added the special tokens, of which <|endoftext|> was not one of them
        # thus we do not need to remove <EOT>
        # (whereas add_special_tokens=True will add this token and we need to remove it to prevent early termination)
        context_labels_length = context_token_ids.shape[1]
        gen_kwargs['decoder_input_ids'] = context_token_ids
        # can inspect what the context looks like with processor.tokenizer.decode(context_token_ids, skip_special_tokens=False)

        pred_ids = model.generate(**inputs, **gen_kwargs)
        # remove the in-context example from the reference using context_labels_length
        pred_ids = pred_ids[:, context_labels_length:]
        generated_text = processor.decode(pred_ids[0], skip_special_tokens=True)
        gt_text_normalised = processor.tokenizer.decode(processor.tokenizer.encode(gt_text), skip_special_tokens=True)
        gt.append(gt_text_normalised)
        # TODO: normalize both texts?
        hyp.append(generated_text)
        print("ref", processor.tokenizer.decode(processor.tokenizer.encode(gt_text), skip_special_tokens=True,
                                         decode_with_timestamps=True))
        print("hyp", processor.decode(pred_ids[0], skip_special_tokens=True, decode_with_timestamps=True))
        print()

    # TODO: note - our timestamp is off by exactly 1 second
    df = Dataset.from_dict({"gt": gt, "hyp": hyp})
    df.to_csv(f"results/icl_{('in' if speech_context else 'ex') + 'clude_speech'}_{run}.csv")
    metrics = jiwer.process_words(gt, hyp)
    print(metrics.wer)
    return metrics.wer

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    SPEECH_CONTEXT = args.speech_context  # True: include audio in the prompt/context
    
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

    dataset_en = load_dataset("google/fleurs", "en_us", cache_dir="/run/user/1000/data")
    dataset_cs = load_dataset("google/fleurs", "cs_cz", cache_dir="/run/user/1000/data")
    dataset_en = dataset_en.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))
    dataset_cs = dataset_cs.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))

    # Limit the samples length in the train split to 15s to be able to easily concatenate them
    for dataset in [dataset_en, dataset_cs]:
        dataset[SPLIT] = dataset[SPLIT].filter(
            lambda x: (x["num_samples"] / x['audio']['sampling_rate']) < MAX_SINGLE_LAN_SEGMENT_LEN)

    # try 10 seeds
    random.seed(12345)
    runs = []
    for run, seed in enumerate([random.randint(10000, 99999) for _ in range(10)]):
        wer = evaluate(
            model, processor, dataset_en[SPLIT], dataset_cs[SPLIT],
            device, seed, SPEECH_CONTEXT, PAUSE,
            run
        )
        runs.append(wer)

    print("WER across", len(runs), "runs:", np.mean(runs), np.std(runs))
    print(runs)
