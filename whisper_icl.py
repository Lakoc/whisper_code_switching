import random
import jiwer
import numpy as np
import torch
from datasets import load_dataset, Audio, Dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


if __name__ == "__main__":
    SPLIT = "validation"
    PAUSE = 1.0
    MAX_SINGLE_LAN_SEGMENT_LEN = 6.0

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

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
        "max_new_tokens": 300, # The length of `decoder_input_ids` equal `prompt_ids` plus special start tokens is 54, and the `max_new_tokens` is 440. Thus, the combined length of `decoder_input_ids` and `max_new_tokens` is: 494. This exceeds the `max_target_positions` of the Whisper model: 448. You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, so that their combined length is less than 448.
        "num_beams": 1,
        "return_timestamps": True,
    }

    # Limit the samples length in the train split to 15s to be able to easily concatenate them
    for dataset in [dataset_en, dataset_cs]:
        dataset[SPLIT] = dataset[SPLIT].filter(
            lambda x: (x["num_samples"] / x['audio']['sampling_rate']) < MAX_SINGLE_LAN_SEGMENT_LEN)

    min_number_of_samples = min(len(dataset_en[SPLIT]), len(dataset_cs[SPLIT]))

    # create new dataset by concatenating samples from two languages
    gt = []
    hyp = []
    for index, (en_sample, cz_sample) in enumerate(zip(dataset_en[SPLIT], dataset_cs[SPLIT])):
        # select in-context example
        in_context_index = index
        while in_context_index == index:
            in_context_index = random.randint(0, min_number_of_samples - 1)
        ic_en, ic_cz = dataset_en[SPLIT][in_context_index], dataset_cs[SPLIT][in_context_index]

        concatenated_sample = torch.from_numpy(np.concatenate(
            [
                # in-context example
                ic_en["audio"]["array"],
                np.zeros(round(ic_en["audio"]["sampling_rate"] * PAUSE)),
                ic_cz["audio"]["array"],
                np.zeros(round(ic_en["audio"]["sampling_rate"] * PAUSE)),
                # target utterance
                en_sample["audio"]["array"],
                np.zeros(round(en_sample["audio"]["sampling_rate"] * PAUSE)),
                cz_sample["audio"]["array"]
             ]))
        inputs = processor(concatenated_sample,
                           sampling_rate=en_sample["audio"]["sampling_rate"],
                           return_tensors="pt",
                           return_attention_mask=True).to(device, dtype=torch_dtype)

        # target utterance
        time_ic_en_end = round(ic_en['num_samples'] / 16_000, 2)
        time_ic_cz_start = time_ic_en_end + PAUSE
        time_ic_cz_end = time_ic_cz_start + round(ic_cz['num_samples'] / 16_000, 2)
        gt_context_readable = f"<|en|><|0.00|>{ic_en['transcription']}<|{time_ic_en_end}|><|cs|><|{time_ic_cz_start}|>{ic_cz['transcription']}<|{time_ic_cz_end}|>"
        # note - Whisper will not duplicate our language tokens
            # but will add the <SOT><no timestamp>
        gt_context = f"<|en|><|0.00|>{ic_en['transcription']}<|{time_ic_en_end}|><|cs|><|{time_ic_cz_start}|>{ic_cz['transcription']}<|{time_ic_cz_end}|>"

        time_en_start = time_ic_cz_end + PAUSE
        time_en_end = time_en_start + round(en_sample['num_samples'] / 16_000, 2)
        time_cz_start = time_en_end + PAUSE
        time_cz_end = time_cz_start + round(cz_sample['num_samples'] / 16_000, 2)
        # adding the timestamp for visual inspection only
        gt_text = f"<|en|><|{time_en_start}|>{en_sample['transcription']}<|{time_en_start}|> <|cs|><|{time_cz_start}|>{cz_sample['transcription']}<|{time_cz_end}|>"

        # set the prefix (in-context text) with decoder_input_ids
        # add_special_tokens=True to make sure we include <|startoftranscript|><|notimestamps|>
        context_token_ids = processor.tokenizer.encode(gt_context, add_special_tokens=True, return_tensors="pt").to(device)
        # remove <EOT> to prevent Whisper from terminating
        context_token_ids = context_token_ids[:, :-1]
        context_labels_length = context_token_ids.shape[1]
        gen_kwargs['decoder_input_ids'] = context_token_ids
        # can inspect what the context looks like with processor.tokenizer.decode(context_token_ids)

        pred_ids = model.generate(**inputs, **gen_kwargs)
        # remove the in-context example from the reference using context_labels_length
        pred_ids = pred_ids[:, context_labels_length:]
        generated_text = processor.decode(pred_ids[0], skip_special_tokens=True)
        gt_text_normalised = processor.tokenizer.decode(processor.tokenizer.encode(gt_text), skip_special_tokens=True)
        gt.append(gt_text_normalised)
        hyp.append(generated_text)
        print(processor.tokenizer.decode(processor.tokenizer.encode(gt_text), skip_special_tokens=True,
                                         decode_with_timestamps=True))
        print(processor.decode(pred_ids[0], skip_special_tokens=True, decode_with_timestamps=True))

    df = Dataset.from_dict({"gt": gt, "hyp": hyp})
    df.to_csv("results_icl.csv")
    metrics = jiwer.compute_measures(gt, hyp)
    print(metrics)
